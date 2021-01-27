# Mamajek
Code to generate mass-distance plots for gravity lenses.

## Dependencies

Can be installed with pip
``` sh
python3 -m pip install -r requirements.txt
```
## How to use

### Mamajek_MCMC

Enter your data in the "input and constants" section. 
Use CLEAN_CUT file from MCMC in .npy format.

Now you can run the code.
``` sh
python3 mamajek_MCMC.py
```

### Mamajek_gauss

Enter your data in the "input and constants" section. 
Use a .txt file in the format as in example below.
```
t0	4818.6367	0.1664
tE	109.5796	4.3268
u0	0.0005	0.05375
piEN	0.0542	0.02555
piEE	0.0251	0.01145
I0	15.1637	0.0002
fs	0.6408	0.04115
```
The second column is the mean value and the third column is the standard deviation.
The columns are separated by a single tab.

Now you can run the code 
``` sh
python3 mamajek_gauss.py
```