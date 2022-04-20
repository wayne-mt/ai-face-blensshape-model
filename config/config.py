class Config(object):
    env = 'default'
    backbone = 'resnet18'
    classify = 'softmax'
    num_classes = 7
    num_classes = 61

    # metric = 'arc_margin'
    metric = ""
    easy_margin = False
    use_se = False
    # loss = 'focal_loss'
    loss = "mlb_reg_loss"

    display = False
    finetune = False

    # train_root = '/home/ubuntu/data/ExpW/data/image_face_aligned'
    # train_list = '/home/ubuntu/data/ExpW/data/label/expw_arcface_pipeline_label.lst'
    # val_list = '/home/ubuntu/data/ExpW/data/label/expw_arcface_pipeline_val_label.lst'
    #
    # test_root = '/data1/Datasets/anti-spoofing/test/data_align_256'
    # test_list = 'test.txt'

    train_root = "/home/ubuntu/data/WeiIphoneCapData"
    # train_list = "/home/ubuntu/data/WeiIphoneCapData/tr_blendshape.lst"
    # val_list = "/home/ubuntu/data/WeiIphoneCapData/ts_blendshape.lst"

    #treat as multi label classification problem
    # train_list = "/home/ubuntu/data/WeiIphoneCapData/tr_blendshape_bce.lst"
    # val_list = "/home/ubuntu/data/WeiIphoneCapData/ts_blendshape_bce.lst"

    #augment the data with more balance ditribution
    train_list = "/home/ubuntu/data/WeiIphoneCapData/tr_blendshape_aug.lst"
    val_list = "/home/ubuntu/data/WeiIphoneCapData/ts_blendshape_aug.lst"

    #
    # checkpoints_path = 'checkpoints'
    # load_model_path = '/home/ubuntu/arcface-pytorch/pretrained/glint360k_cosface_r18_fp16_0.1/backbone.pth'
    # test_model_path = 'checkpoints/resnet18_110.pth'

    checkpoints_path = 'checkpoints_blend'
    load_model_path = 'checkpoints/resnet18_40.pth'
    test_model_path = 'checkpoints/resnet18_110.pth'

    save_interval = 10

    train_batch_size = 64  # batch size
    test_batch_size = 60

    input_shape = (1, 128, 128)

    optimizer = 'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '0'
    num_workers = 4  # how many workers for loading data
    print_freq = 100  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 200
    lr = 1e-1  # initial learning rate
    lr_step = 50
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
