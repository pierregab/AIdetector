from function_to_test import test_model

# Call the test function with your model and data paths
test_model('final_model.keras', 'H01_labelCNN_50x50grid_RAWPM.bin', num_samples=100000, grid_size=16)
