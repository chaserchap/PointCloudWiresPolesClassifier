# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import sys

load_dotenv(find_dotenv(), verbose=True)
os.chdir('/home/chapmaca/Projects/pointcloudclassifier')
print(os.getcwd())
sys.path.append('/home/chapmaca/Projects/pointcloudclassifier')
from src.classes.PDALObject import PDALObject
from references.resources import test_pipe


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    input_filetype = Path(input_filepath).suffix.lower()
    pdal_supported_filetypes = ['.las', '.laz', '.npy', '.bpf']
    if input_filetype == '.las' or input_filetype == '.laz':
        logger.info('{0} filetype supported by pdal'.format(input_filetype))
        interim_filepath = input_filepath
    else:
        logger.info('input filetype not supported by pdal. attempting to read in file as csv')
        for delim in [',', ' ', '\t', '|']:
            data = pd.read_csv(input_filepath, sep=delim, names=['X', 'Y', 'Z'])
            if data.shape[1] >= 3:
                break
        interim_filepath = Path(project_dir).joinpath('data', 'interim', Path(input_filepath).stem + '.npy')
        logger.info('saving interim file to {0}'.format(interim_filepath))
        np.save(interim_filepath, data.to_records(index=False))

    logger.info('establishing PDALObject')
    po = PDALObject(interim_filepath)

    logger.info('setting json pipeline: {0}'.format(test_pipe))
    po.set_json_pipeline(test_pipe)

    logger.info('setting output filepath to: {0}'.format(output_filepath))
    po.outfile = output_filepath

    logger.info('running pdal pipeline')
    po.execute()



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
  

    main()
