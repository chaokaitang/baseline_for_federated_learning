import pickle

# 改成你自己的路径
train_path = "E:\\Desktop\\graduate_year_project\\initial\\code\\baseline_test\\data\\emnist\\data\\train\\emnist_balanced_0_shard2_niid.pkl"
test_path  = "E:\\Desktop\\graduate_year_project\\initial\\code\\baseline_test\\data\\emnist\\data\\test\\emnist_balanced_0_shard2_niid.pkl"

with open(train_path, "rb") as f:
    train_data = pickle.load(f)

with open(test_path, "rb") as f:
    test_data = pickle.load(f)

# 随便选一个 client，比如 0
cid = train_data['users'][61]

print("Client ID:", cid)

print("Train samples:", len(train_data['user_data'][cid]['y']))
print("Test samples :", len(test_data['user_data'][cid]['y']))

print("Train labels:", set(train_data['user_data'][cid]['y']))
print("Test labels :", set(test_data['user_data'][cid]['y']))
