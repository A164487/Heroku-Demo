{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "import pickle\n",
    "\n",
    "from sklearn.cluster import MiniBatchKMeans"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('Clean_data.csv')\n",
    "X = df.iloc[:,:16]\n",
    "y = df.iloc[:, -1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "mbk = MiniBatchKMeans(init ='k-means++', n_clusters = 3, batch_size = 100, n_init = 10, max_no_improvement = 10, verbose = 0) \n",
    "\n",
    "mbk.fit(X, y)\n",
    "\n",
    "pickle.dump(mbk, open('model.pkl','wb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1]\n"
     ]
    }
   ],
   "source": [
    "model = pickle.load(open('model.pkl','rb'))\n",
    "print(model.predict([[2,30,9,0,0,0,1,1,0,0,0,1,1,1,1256,1256]]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1]\n"
     ]
    }
   ],
   "source": [
    "model = pickle.load(open('model.pkl','rb'))\n",
    "print(model.predict([[2,20,0,0,0,0,1,1,0,0,0,1,1,1,1256,0]]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0]\n"
     ]
    }
   ],
   "source": [
    "model = pickle.load(open('model.pkl','rb'))\n",
    "print(model.predict([[2,20,7,0,0,0,0,1,0,0,0,2,3,1,5076,1692]]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0]\n"
     ]
    }
   ],
   "source": [
    "model = pickle.load(open('model.pkl','rb'))\n",
    "print(model.predict([[2,10,9,0,0,0,0,1,0,0,0,1,1,1,5972,1256]]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[2]\n"
     ]
    }
   ],
   "source": [
    "model = pickle.load(open('model.pkl','rb'))\n",
    "print(model.predict([[2,10,9,0,0,0,0,1,0,0,0,9,1,1,22157,1256]]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0]\n"
     ]
    }
   ],
   "source": [
    "model = pickle.load(open('model.pkl','rb'))\n",
    "print(model.predict([[1,40,4,0,0,0,1,1,1,0,0,1,1,1,3801,2109]]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[2]\n"
     ]
    }
   ],
   "source": [
    "model = pickle.load(open('model.pkl','rb'))\n",
    "print(model.predict([[1,40,7,0,0,0,0,1,1,0,0,1,1,1,29101,1692]]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[2]\n"
     ]
    }
   ],
   "source": [
    "model = pickle.load(open('model.pkl','rb'))\n",
    "print(model.predict([[2,70,2,0,1,1,1,1,1,1,1,1,2,6,19876,8554]]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[2]\n"
     ]
    }
   ],
   "source": [
    "model = pickle.load(open('model.pkl','rb'))\n",
    "print(model.predict([[2,70,2,0,1,1,1,1,1,1,1,1,2,1,19876,8554]]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0]\n"
     ]
    }
   ],
   "source": [
    "model = pickle.load(open('model.pkl','rb'))\n",
    "print(model.predict([[2,70,8,0,0,0,0,1,0,0,0,2,4,6,5876,1654]]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
