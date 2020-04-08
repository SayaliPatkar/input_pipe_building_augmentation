def get_model_file(path_net_pb):
    if os.path.exists(path_net_pb) and path_net_pb.endswith('.pb'):
        model_dir = path_net_pb
        print("Loading the model from {}".format(model_dir))
    elif os.path.exists(path_net_pb) and os.path.isdir(path_net_pb):
        file = glob(os.path.join(path_net_pb, '*.pb'))
        if file != [] and len(file) == 1:
            model_dir = path_net_pb
            print("Loading the model from {}".format(model_dir))
        elif not file != []:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file)
        else:
            raise CustomError(f"Directory '{path_net_pb}' has multiple model files.")
    else:
        raise NotADirectoryError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), path_net_pb)
    return model_dir
