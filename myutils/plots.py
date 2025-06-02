# from pathlib import Path
import matplotlib.pyplot as plt
import torch
# import torch.nn as nn
import os

class Feature_vis():
    def __init__(self, model, save_dir, logging=None):
        self.model = model
        self.save_dir = save_dir
        self.logging = logging
        # self.features = []  # 用于存储每层的特征图
        self.cls_id = {}
        self._register_hooks()
        self.ignore = ['GlobalQueryGen', 'WindowAttention', 'WindowAttentionGlobal']
        self.required = ['ReduceSize', 'PatchEmbed', 'Mlp', 'GCViTBlock', 'GCViTLayer', 'UpSize', 'Up_GCViTLayer', ]
        
    def reset(self):
        self.cls_id = {}    
        
    def _register_hooks(self):
        # def hook_fn(module, input, output):
        #     # 将每一层的输出（特征图）保存下来
        #     self.features.append(output)
        
        # 注册所有卷积层的钩子
        # print(self.model.children())
        for layer in self.model.modules():
            # print(type(layer))
            if layer.__module__.startswith('networks'):
                # print(type(layer))
                # print(layer.__name__)
                layer.register_forward_hook(self.feature_visualization)        
        
    def feature_visualization(self, module, inp, out):
        # n = 32
        
        
        class_name = str(type(module)).split('.')[-1].rstrip("'>")
        # print(class_name)
        self.cls_id[class_name] = self.cls_id.get(class_name, 0) + 1        
        
        if class_name == 'FreqFusion':
            out = out[-1]
        
        if class_name not in self.ignore:
            if class_name in self.required:
                out = out.permute(0, 3, 1, 2)
                
            # print(class_name)
            # print(out.shape)
            batch, channels, height, width = out.shape  # batch, channels, height, width
            if height > 1 and width > 1:
                # print(module.__module__.split('.')[-1])
                
                f = os.path.join(self.save_dir, f"{class_name}[{self.cls_id.get(class_name)}]_features.png")
                # f = self.save_dir / f"{class_name}[{cls_id.get(class_name)}]_features.png"  # filename        
                out = torch.mean(out[0].detach().cpu(), dim=0)
                # blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
                # n = min(n, channels)  # number of plots
                # fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8 rows x n/8 cols
                # ax = ax.ravel()
                # plt.subplots_adjust(wspace=0.05, hspace=0.05)
                # for i in range(n):
                #     ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
                #     ax[i].axis('off')
                plt.figure(figsize=(6.4, 6.4))
                plt.axis('off')
                plt.imshow(out)
                if self.logging:
                    self.logging.info(f'Saving {f}')
                plt.savefig(f, dpi=300, bbox_inches='tight')
                plt.close()
            # np.save(str(f.with_suffix('.npy')), x[0].cpu().numpy())  # npy save
        # print(self.cls_id)

