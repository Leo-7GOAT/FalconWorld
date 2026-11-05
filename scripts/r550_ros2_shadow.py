"""ROS2 Humble shadow node: subscribes Odometry, publishes no drive commands.

The installed hardware driver, topics, steering semantics and emergency stop
must be verified separately before developing an explicitly authorized actuator
transport. This node deliberately has no motor publisher or motion switch.
"""
import argparse
import json
import math
import time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from falconworld.adapters import R550Config,R550ShadowAdapter
from falconworld.contracts import Action,Observation


def main():
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import TwistStamped
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--target-speed',type=float,default=.3)
    p.add_argument('--output',default='outputs/r550/ros_shadow.jsonl')
    p.add_argument('--model',help='LowSpeedModel NPZ; simulation-trained models remain UNCALIBRATED')
    p.add_argument('--command-reference-topic',help='Verified TwistStamped stream of actual SENT speed references, not model recommendations')
    args,ros_args=p.parse_known_args()
    if not math.isfinite(args.target_speed) or args.target_speed<0:
        p.error('--target-speed must be finite and nonnegative')
    config=R550Config.load(args.config);adapter=R550ShadowAdapter(config)
    planner=None
    if args.model:
        if not args.command_reference_topic:
            p.error('--model requires an explicit verified --command-reference-topic; previous command cannot be inferred from stationary feedback')
        from falconworld.r550_worldmodel import LowSpeedModel,LowSpeedMPC
        planner=LowSpeedMPC(LowSpeedModel.load(args.model))
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    rclpy.init(args=ros_args)
    class Shadow(Node):
        def __init__(self):
            super().__init__('falconworld_shadow')
            self.last=None;self.last_received=None;self.previous_speed=0.
            self.references=[];self.model_history_valid=False
            self.log=output.open('a',encoding='utf-8')
            self.create_subscription(Odometry,config.odometry_topic,self.feedback,10)
            if planner is not None:
                self.create_subscription(TwistStamped,args.command_reference_topic,self.command_reference,10)
            self.create_timer(min(.05,config.command_timeout_s/2),self.tick)

        def feedback(self,msg):
            now=self.get_clock().now().nanoseconds/1e9
            stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
            v=msg.twist.twist.linear.x
            if not all(math.isfinite(x) for x in (v,stamp)) or stamp<0 or v<0:
                self.get_logger().error('Invalid forward feedback; shadow stopped');self.last=None;return
            if self.last is not None and stamp<=self.last.timestamp:
                self.get_logger().error('Nonmonotonic feedback; shadow stopped');self.last=None;return
            self.model_history_valid=self.last is not None and .0475<=stamp-self.last.timestamp<=.0525
            self.previous_speed=v if self.last is None else self.last.speed
            self.last=Observation(stamp,v);self.last_received=now

        def command_reference(self,msg):
            stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
            v=msg.twist.linear.x
            if not math.isfinite(stamp) or stamp<0 or not 0<=v<=min(.3,config.speed_limit_m_s):
                self.references=[];return
            if self.references and stamp<=self.references[-1][0]:
                self.references=[];return
            self.references=(self.references+[(stamp,v)])[-128:]

        def tick(self):
            now=self.get_clock().now().nanoseconds/1e9
            if self.last is None:
                record=dict(timestamp=now,mode='shadow',target_speed_m_s=0,sent=False,acknowledged=False,reason='missing_feedback')
            elif planner is None:
                # PID demonstration only; acceleration never sent as velocity.
                accel=1.2*(min(args.target_speed,config.speed_limit_m_s)-self.last.speed)
                record=adapter.shadow(self.last,Action(accel),now)
            else:
                reason=''
                causal=[r for r in self.references if r[0]<=self.last.timestamp]
                if not causal or not 0<=now-causal[-1][0]<=config.feedback_timeout_s:
                    reason='missing_or_stale_actual_command_reference'
                elif not self.model_history_valid:
                    reason='feedback_period_mismatch'
                elif not 0<=now-self.last.timestamp<=config.feedback_timeout_s:
                    reason='missing_or_stale_feedback'
                elif max(self.last.speed,self.previous_speed)>.35:
                    reason='model_input_out_of_domain'
                if reason:
                    record=dict(timestamp=now,mode='worldmodel_shadow',target_speed_m_s=0.,reason=reason,sent=False,acknowledged=False)
                else:
                    begin=time.perf_counter()
                    command,info=planner.plan([self.last.speed,self.previous_speed,causal[-1][1]],
                                              min(args.target_speed,config.speed_limit_m_s,.3))
                    elapsed=time.perf_counter()-begin
                    if elapsed>config.command_timeout_s:command=0.;reason='planner_timeout'
                    record=dict(timestamp=now,mode='worldmodel_shadow',target_speed_m_s=command,
                                reason=reason or info['reason'],cycle_ms=elapsed*1000,
                                model_origin=planner.model.origin,sent=False,acknowledged=False)
            self.log.write(json.dumps(record)+'\n');self.log.flush()

    node=Shadow()
    try:rclpy.spin(node)
    finally:node.log.close();node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
