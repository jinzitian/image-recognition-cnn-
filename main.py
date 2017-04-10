# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 00:35:19 2017

@author: Jinzitian
"""

import sys

from predict import image_process,predict
from train import train

def main(args):
    
    if args[1] == 'train':
        train()
    elif args[1] == 'predict':
        try:
            image = image_process(args[2])
            class_label = predict(image)
            print('\nThe predict class is: %s\n'%class_label)
        except Exception as e:
            print(e)
            print('\nplease python main.py predict ./image/make_id/model_id/released_year/xxxxx.jpg\n')
            
    else:
        print('please try:')
        print('1, python main.py train') 
        print('2, python main.py predict ./image/make_id/model_id/released_year/xxxxx.jpg')
        
if __name__=='__main__':
    
    main(sys.argv)