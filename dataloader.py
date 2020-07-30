###########################
## Author - Shuting Xing ##
## The following function is for loading and preparing the data for Model



def dataloader_cnn(data_args_list):
    import pandas as pd;
    import numpy as np;

    train_data = pd.read_csv(data_args_list['data_path'] + 'train/train.csv')
    test_data = pd.read_csv(data_args_list['data_path'] + 'test/test.csv')
    val_data = pd.read_csv(data_args_list['data_path'] + 'val/val.csv')

    x_train = train_data.drop('label',axis=1)
    y_train = train_data['label']
    x_test = test_data.drop('label',axis=1)
    y_test = test_data['label']
    x_val = val_data.drop('label',axis=1)
    y_val = val_data['label']

    img_x, img_y = 28,28

    x_train = np.array(x_train).reshape(-1, img_x, img_y, 1)
    x_test = np.array(x_test).reshape(-1, img_x, img_y, 1)   
    x_val = np.array(x_val).reshape(-1, img_x, img_y, 1)   
    
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_val = x_val.astype('float32')
    
    x_train /= 255
    x_test /= 255
    x_val /= 255
    
    num_classes = 10
    input_shape = (img_x, img_y, 1)

    print('x_train: ', x_train.shape)
    print('y_train', y_train.shape)
    print('x_test: ', x_test.shape)
    print('y_test', y_test.shape)
    print('x_val: ', x_val.shape)
    print('y_val', y_val.shape)
    print('input_shape: ', input_shape )

    return x_train, x_test, x_val, y_train, y_test, y_val

def dataloader_xgb(data_args_list):
    import pandas as pd;
    import numpy as np;
    train_data = pd.read_csv(data_args_list['data_path'] + 'train/train.csv')
    test_data = pd.read_csv(data_args_list['data_path'] + 'test/test.csv')
    val_data = pd.read_csv(data_args_list['data_path'] + 'val/val.csv')

    x_train = train_data.drop('label',axis=1)
    y_train = train_data['label']
    x_test = test_data.drop('label',axis=1)
    y_test = test_data['label']
    x_val = val_data.drop('label',axis=1)
    y_val = val_data['label']

    print('x_train: ', x_train.shape)
    print('y_train', y_train.shape)
    print('x_test: ', x_test.shape)
    print('y_test', y_test.shape)
    print('x_val: ', x_val.shape)
    print('y_val', y_val.shape)

    return x_train, x_test, x_val, y_train, y_test, y_val
