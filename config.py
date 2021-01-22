from optparse import OptionParser


parser = OptionParser()

parser.add_option('-e', '--epochs', dest='epochs', default=200, type='int',
                  help='number of epochs (default: 80)')
parser.add_option('-b', '--batch-size', dest='batch_size', default=10, type='int',
                  help='batch size (default: 16)')
parser.add_option('--df', '--disp_freq', dest='disp_freq', default=20, type='int',
                  help='frequency of displaying the training results (default: 100)')
parser.add_option('--vf', '--val_freq', dest='val_freq', default=60, type='int',
                  help='run validation for each <val_freq> iterations (default: 2000)')
parser.add_option('-j', '--workers', dest='workers', default=16, type='int',
                  help='number of data loading workers (default: 16)')

# For data
parser.add_option('--dn', '--data_name', dest='data_name', default='Glum',
                  help='mnist, fashion_mnist, t_mnist, c_mnist (default: mnist)')
parser.add_option('--dl', '--data_len', dest='data_len', default=None, type='int',
                  help='Number of data samples (default: None which automatically takes all samples)')
parser.add_option('--ih', '--img_h', dest='img_h', default=256, type='int',
                  help='input image height (default: 256)')
parser.add_option('--iw', '--img_w', dest='img_w', default=256, type='int',
                  help='input image width (default: 256)')
parser.add_option('--ic', '--img_c', dest='img_c', default=3, type='int',
                  help='number of input channels (default: 3)')

# Leave one out set (LOOS)
parser.add_option('--loos', '--loos', dest='loos', default=4, type='int',
                  help='index of the set to leave out for testing (default: 4)')

# Classes
parser.add_option('--nc', '--num_classes', dest='num_classes', default=2, type='int',
                  help='number of classes (default: 5)')

# GAN training activation
parser.add_option('--gan', '--gan_training', dest='GAN', default=1, type='int',
                  help='Train with or without GAN augmented dataset (default 0, no GAN)')

# GAN count per class
parser.add_option('--num_gan', '--num_gan', dest='num_gan', default=10000, type='int',
                  help='Choose how many Style Transfer images to augment your set with')

# Uncertainty Fraction
parser.add_option('--uf', '--unc_frac', dest='unc_frac', default=0.52, type='float',
                  help='uncertainty fraction of data (default: 0.52)')

# For model
parser.add_option('--m', '--model', dest='model', default='densenet',
                  help='vgg, inception, resnet, densenet (default: resnet)')
parser.add_option('--lr', '--lr', dest='lr', default=0.001, type='float',
                  help='learning rate(default: 0.001)')

# For directories
parser.add_option('--sd', '--save-dir', dest='save_dir', default='./save',
                  help='saving directory of .ckpt models (default: ./save)')

parser.add_option('--gc', '--gradcam', dest='gradcam', default=True,
                  help='whether to run gradcam when running inference or not (default: True)')

# MC Dropout Uncertainty

parser.add_option('--mc', '--MC', dest='MC', default=False,
                  help='whether to run MC simulation or not (default: True)')

parser.add_option('--mci', '--mc_iter', dest='mc_iter', default=60, type='int',
                  help='number of MC iterations(default: 100)')

parser.add_option('--mcdir', '--mc_dir', dest='mc_dir',
                  default='./mc_results',
                  help='path to save MC-Dropout uncertainty predictions')

parser.add_option('--lp', '--load_model_path', dest='load_model_path',
                  default='<your_model_path>',
                  help='path to load a .ckpt model')



options, _ = parser.parse_args()

