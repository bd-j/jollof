#!/usr/bin/env python3
import argparse
from bens_original import *


#######################################
# Create command line argument parser
#######################################

def create_parser():

        # Handle user input with argparse
    parser = argparse.ArgumentParser(
        description="Mock LF flags and options from user.")

    parser.add_argument('--logphistar',
        dest='logphistar',
        default=-4,
        metavar='logphistar',
        type=float,
        help='Log10 phistar.')

    parser.add_argument('--mstar',
        dest='mstar',
        default=-19,
        metavar='mstar',
        type=float,
        help='mstar.')

    parser.add_argument('--alpha',
        dest='alpha',
        default=-2,
        metavar='alpha',
        type=float,
        help='alpha.')

    parser.add_argument('-v', '--verbose',
        dest='verbose',
        action='store_true',
        help='Print helpful information to the screen? (default: False)',
        default=False)

    return parser

#######################################
# main function
#######################################
def main():
	#create the command line argument parser
	parser = create_parser()

	#store the command line arguments
	args   = parser.parse_args()

	#produce the mock lf
	process_mocks(logphistar=args.logphistar,mstar=args.mstar,alpha=args.alpha)

#######################################
# run the program
#######################################
if __name__=="__main__":
	main()
