### command to run
```bash
python -m exp.train_model -F test_runs with experiments/train_model/best_runs/Spheres/TopoRegEdgeSymmetric.json device='cuda'   
``` 
### pipeline
`train_model.py`:
=>  from .ingredients import model as model_config
    from .ingredients import dataset as dataset_config
    Get data and model 
    Run training via `training_loop()`, constructed from `train.py`
    Save model and state

`exp.ingredients.model.py`:
=>  `from src import models`
    Get the model class via `model_cls = getattr(models, name)`

`src.models`
    `.__init__.py`:
        Names of all models
    `approx_based.py`:
        from src.topology import PersistentHomologyCalculation 
        from src.models import submodules
        from src.models.base import AutoencoderModel
    =>  class TopologicallyRegularizedAutoencoder(AutoencoderModel)
        class PersistentHomologyCalculation
        class TopologicalSignatureDistance(nn.Module)
    `submodules.py`:
    =>  Subclass of AutoencoderModel
        e.g. class ConvolutionalAutoencoder(AutoencoderModel)
    `base.py`:
        class AutoencoderModel(nn.Module, metaclass=abc.ABCMeta):
            @abc.abstractmethod
            def forward(self, x) -> Tuple[float, Dict[str, float]]:
            def encode(self, x)
            def decode(self, z)

### utilization
=>  approx_based ~ autoencoder_model: TopologicallyRegularizedAutoencoder 
=>  sub ~ autoencoders: ConvolutionalAutoencoder ~ SwinUnet (SwinTransformerSys)
                    init, forward, encoder, decoder
=>  base: Autoencoder
    vanilla: SwinUnetAutoencoderModel
    
