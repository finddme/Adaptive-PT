# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
from pymongo import MongoClient
import json
from collections import OrderedDict
import logging
from torch.utils.data import Dataset

class Mongo(Dataset):
    def __init__(self, mongo_uri, db_name, collection):
        self.db_name = db_name
        self.collection = collection
        self.mongo_uri = mongo_uri
        self.client = MongoClient(self.mongo_uri)


    def find_item(self, condition = None, data = None):
        result = self.client[self.db_name][self.collection].find({})
        return result



