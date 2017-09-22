# Pi-NN
## Note
Please check https://github.com/Oscarlight/PiNN_Caffe2 for easy-to-use device compact modeling platform based on Caffe2.

## Reference
Please refer to the paper "Physics-Inspired Neural Networks (Pi-NN) for Efficient Device Compact Modelling" for the details of the model. (Link to the paper: http://ieeexplore.ieee.org/document/7778193/)

## How to use:
- **Training**: dmm.ml, inout.ml, and util.ml are the source codes for traning Pi-NN written in (take a deep breath) OCaml. The main routine is in "dmm.ml". It will read the training/testing data from the *data* directory and output the trained model and log to the *model* directory. The executable nfit can be compile by:

```shell
ocamlopt str.cmxa util.ml inout.ml dmm.ml -o nfit
rm *.cmi *.o *.cmx
./nfit
```
- **Ploting**: The python source codes in the *plot* directory are used to produce **Fig 6 c-f** in the paper. The main routine for ploting is in "pinn.py". The *original_data* directory contains the original data without pre-processing (i.e. shift and scale).
