import os
import sys
import torch

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 现在可以导入项目模块
from datasets.loader import construct_loader
from utils.parsing import parse_train_args
from utils.diffusion_utils import t_to_sigma


def check_graph(graph, idx=None):
    """检查图数据中的候选边是否存在问题"""
    # 检查候选边是否存在
    if ('ligand', 'nci_cand', 'receptor') not in graph.edge_types:
        return

    edge = graph['ligand', 'nci_cand', 'receptor']
    if not hasattr(edge, 'edge_index'):
        return

    edge_index = edge.edge_index
    num_lig = graph['ligand'].num_nodes
    num_rec = graph['receptor'].num_nodes

    # 1) 检查 edge_index 越界
    bad_lig = (edge_index[0] < 0) | (edge_index[0] >= num_lig)
    bad_rec = (edge_index[1] < 0) | (edge_index[1] >= num_rec)
    if bad_lig.any() or bad_rec.any():
        print(f"[BAD edge_index] graph={idx} num_lig={num_lig} num_rec={num_rec} "
              f"bad_lig={bad_lig.nonzero().flatten().tolist()} "
              f"bad_rec={bad_rec.nonzero().flatten().tolist()}")
        return

    # 2) 检查 edge_type_y 范围
    if hasattr(edge, 'edge_type_y'):
        labels = edge.edge_type_y
        if (labels < 0).any() or (labels >= 8).any():
            bad = ((labels < 0) | (labels >= 8)).nonzero().flatten().tolist()
            print(f"[BAD labels] graph={idx} bad_idx={bad} "
                  f"min={int(labels.min())} max={int(labels.max())}")


def run(dataset):
    """运行数据集检查"""
    for i in range(len(dataset)):
        g = dataset.get(i)
        check_graph(g, idx=i)


def main():
    """主函数"""
    # 解析参数
    args = parse_train_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # t_to_sigma 函数已经定义在 diffusion_utils 中，可以直接使用
    # 它接受 t_tr, t_rot, t_tor 和 args 作为参数
    
    try:
        # 构建数据加载器
        train_loader, val_loader, _ = construct_loader(args, t_to_sigma, device)
        
        # 检查训练数据集
        print(f"开始检查训练数据集，共 {len(train_loader.dataset)} 个样本")
        run(train_loader.dataset)
        print("训练数据集检查完成")
        
        # 也可以选择检查验证数据集
        if val_loader is not None:
            print(f"开始检查验证数据集，共 {len(val_loader.dataset)} 个样本")
            run(val_loader.dataset)
            print("验证数据集检查完成")
            
    except Exception as e:
        print(f"检查过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
