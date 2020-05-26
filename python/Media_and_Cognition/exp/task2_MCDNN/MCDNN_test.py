from MCDNN import Model
test_path = './Test/'
classes = ('i2', 'i4', 'i5', 'io', 'ip', 'p11', 'p23', 'p26', 'p5', 'pl30',
               'pl40',  'pl5', 'pl50', 'pl60', 'pl80', 'pn', 'pne', 'po', 'w57')
config_dict = {'batch_size':100,
               'lr':0.001,
              'is_train':False,
               'epoch_num':40,
               'validation_epochs':5}
model = Model(data_path=test_path, classes=classes, config=config_dict)
model.load_best_model()
model.Test()
