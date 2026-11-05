import unittest
from falconworld.contracts import History,Observation,Action
from falconworld.bridges import observed_history,SensorWorldModel
from falconworld.models import PhysicsModel


class BridgeTests(unittest.TestCase):
    def test_missing_required_sensor_and_sampling(self):
        history=History();history.append(Observation(0,2))
        with self.assertRaises(ValueError):observed_history(history)
        history=History();history.append(Observation(0,2,gap=10,lead_speed=2))
        history.append(Observation(.05,2,gap=10,lead_speed=2))
        with self.assertRaises(ValueError):observed_history(history)

    def test_causal_sensor_and_action_conditioning(self):
        history=History();history.append(Observation(0,2,gap=10,lead_speed=2))
        model=SensorWorldModel(PhysicsModel())
        a=model.predict(history,[Action(2)]*10)
        b=model.predict(history,[Action(-4)]*10)
        self.assertGreater(a[-1].speed,b[-1].speed)
        self.assertAlmostEqual(a[-1].timestamp,1.)
