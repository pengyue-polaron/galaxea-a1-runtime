import pickle

with open("/home/pengyue/Codespace/DataCoach/data/processed_data/pick_twice/demo_0/states.pkl", "rb") as f:
    states = pickle.load(f)

with open("/home/pengyue/Codespace/DataCoach/data/processed_data/pick_twice/demo_0/commanded_states.pkl", "rb") as f:
    commanded_states = pickle.load(f)
 
print("states:")   
print(states[0])
print(states[0].keys()) # dict_keys(['timestamp', 'data'])
print(states[0]["data"].keys()) # dict_keys(['timestamp', 'pos', 'ori', 'joint', 'gripper', 'gripper_source'])

print("commanded_states:")
print(commanded_states[0])
print(commanded_states[0].keys()) # dict_keys(['timestamp', 'data'])
print(commanded_states[0]["data"].keys()) # dict_keys(['timestamp', 'pos', 'ori', 'joint', 'gripper', 'gripper_source'])