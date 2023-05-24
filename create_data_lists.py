from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(train_folders=['./data/train_CelebA_128'],
                      test_folders=['./data/test_CelebA_128'],
                      verify_folders=['./data/verify_CelebA_128'],
                      min_size=100,
                      output_folder='./data/')
