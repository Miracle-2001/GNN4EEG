Here, we present the whole progress to download and preprocess the FACED dataset. Please follow the steps below:

1. Download the **Processed_data.zip** according to the link: https://www.synapse.org/#!Synapse:syn50615881

2. Download the **Code.zip**.

3. Further process the data according to the steps provided in the *Readme.md* file under the **Code.zip** file.


For step 3, You can conduct similar commands like the first 3 steps in the *SVM_analysis* chapter of that *Readme.md* file, which are:


1. DE feature calculation 

(Note that **this script only calculate and store the theta, alpha, beta, and gamma frequency band**, so if you are also interested in the feature extracted from the delta band, you should modify the *save_de.py* file on your own)
```bash
python save_de.py
```

2. Running_norm calculation 

(--n-vids 24 for binary classification; --n-vids 28 for nine-category classification, same below)

```bash
python running_norm.py --n-vids 24   
```

3. Using LDS to smooth the data 
```bash
python smooth_lds.py --n-vids 24
```

<!-- ---

However, 
If you think,  the procedure is still cumbersome, inflexible and boring, never mind. We also offer you a easy-to-use script for data preprocessing which can be found in $\color{red}{TODO}$ -->