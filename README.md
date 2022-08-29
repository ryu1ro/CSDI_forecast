# CSDI_forecast
This is the github repository for the dissertaiton. 
This model provides time series forecast based on the CSDI model.

## Dataset
Souce: https://github.com/awslabs/gluon-ts.git (Gluon-ts) \
The data downloaded from Gluon-ts is organised using the following code.\
make_dataset.ipynb

## Execution on `Solar` dataset
```bash
python CSDI_exe.py --'solar'
```

## Refference
This implementation is based on 
- https://github.com/ermongroup/CSDI.git (CSDI)
- https://github.com/ermongroup/ddim.git (DDIM sampling)
- https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/mlp/mlp_mixer.py (MLP-Mixer)
