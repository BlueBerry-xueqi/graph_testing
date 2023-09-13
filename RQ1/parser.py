import argparse


class Parser:

    def __init__(self):
        # create a parser
        self.parser = argparse.ArgumentParser(description='Graph Multiset Transformer')
        self.set_arguments()

    # add arguments
    def set_arguments(self):
        self.parser.add_argument('--epochs', default=20, type=int, help='train epochs number')
        self.parser.add_argument('--TU', default=False, type=bool, help='whether use TU models')
        self.parser.add_argument("--retrain_epochs", type=int, default=3, dest="retrain_epochs")
        self.parser.add_argument('--type', default='GCN', type=str,
                                 choices=["GIN", "GMT", "SAGE", "GraphNN", "NN", "GCN", "GAT", "ARMA", "AGNN",
                                          "mempool", "GINJK", "unet", "GINE", "gra", "diff"], required=False)
        self.parser.add_argument('--data', type=str, default='Cora',
                                 help='dataset type')

        self.parser.add_argument('--select_ratio', type=int, default=1,
                                 help='select dataset num', dest='select_ratio')

        self.parser.add_argument('--metrics', type=str, default="random",
                                 choices=["deepgini", "random", "l_con","entropy", "margin", "kmeans", "MCP", "bald", "k-center","variance", "CES", "spec", "GMM", "Hierarchical"])
        self.parser.add_argument("--model", type=str, default='GMT', choices=['GMT', 'voxel'])

        self.parser.add_argument("--model-string", type=str, default='GMPool_G-SelfAtt-GMPool_I')

        self.parser.add_argument('--conv', default='GCN', type=str,
                                 choices=['GCN', 'GIN'],
                                 help='message-passing function type')
        self.parser.add_argument('--num-convs', default=3, type=int)
        self.parser.add_argument('--mab-conv', default='GCN', type=str,
                                 choices=['GCN', 'GIN'],
                                 help='Multi-head Attention Block, GNN type')
        self.parser.add_argument('--seed', type=int, default=42, help='seed')

        self.parser.add_argument('--num-hidden', type=int, default=128, help='hidden size')
        self.parser.add_argument('--num-heads', type=int, default=1, help='attention head size')

        self.parser.add_argument('--batch-size', default=128, type=int, help='train batch size')
        self.parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
        self.parser.add_argument('--weight-decay', type=float, default=0.0001, help='weight decay')
        self.parser.add_argument("--grad-norm", type=float, default=1.0)
        self.parser.add_argument("--dropout", type=float, default=0.5)

        self.parser.add_argument('--pooling-ratio', type=float, default=0.25, help='pooling ratio')

        self.parser.add_argument("--gpu", type=int, default=-1)
        self.parser.add_argument('--patience', type=int, default=50, help='patience for earlystopping')

        self.parser.add_argument("--ln", action='store_true')
        self.parser.add_argument("--lr-schedule", action='store_true')
        self.parser.add_argument("--cluster", action='store_true')
        self.parser.add_argument("--normalize", action='store_true')

        self.parser.add_argument('--exp', default='1', choices=[1, 2, 3], type=int, dest='exp')
        self.parser.add_argument('--savedpath', default="saved", type=str)

    def parse(self):
        args, unparsed = self.parser.parse_known_args()

        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))

        return args