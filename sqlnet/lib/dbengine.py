# -*- coding:utf-8 -*-
import json
import records
import re

agg_dict = {0:"", 1:"AVG", 2:"MAX", 3:"MIN", 4:"COUNT", 5:"SUM"}
cond_op_dict = {0:">", 1:"<", 2:"==", 3:"!="}
rela_dict = {0:'', 1:' AND ', 2:' OR '}

class DBEngine:
    def __init__(self, fdb):
        self.db = records.Database('sqlite:///{}'.format(fdb))
        self.conn = self.db.get_connection()

    def execute(self, table_id, select_index, aggregation_index, conditions, condition_relation):
        """
        table_id: id of the queried table.
        select_index: list of selected column index, like [0,1,2]
        aggregation_index: list of aggregation function corresponding to selected column, like [0,0,0], length is equal to select_index
        conditions: [[condition column, condition operator, condition value], ...]
        condition_relation: 0 or 1 or 2
        """
        table_id = 'Table_{}'.format(table_id)

        # 条件数>1 而 条件关系为''
        if condition_relation == 0 and len(conditions) > 1:
            return 'Error1'
        # 选择列或条件列为0
        if len(select_index) == 0 or len(conditions) == 0 or len(aggregation_index) == 0:
            return 'Error2'

        condition_relation = rela_dict[condition_relation]

        select_part = ""
        for sel, agg in zip(select_index, aggregation_index):
            select_str = 'col_{}'.format(sel+1)
            agg_str = agg_dict[agg]
            if agg:
                select_part += '{}({}),'.format(agg_str, select_str)
            else:
                select_part += '({}),'.format(select_str)
        select_part = select_part[:-1]

        where_part = []
        for col_index, op, val in conditions:
            if type(val) == unicode:
                where_part.append('col_{} {} "{}"'.format(col_index+1, cond_op_dict[op], val.encode('utf-8')))
            else:
                where_part.append('col_{} {} "{}"'.format(col_index+1, cond_op_dict[op], val))
        where_part = 'WHERE ' + condition_relation.join(where_part)

        query = 'SELECT {} FROM {} {}'.format(select_part, table_id, where_part)
        query = query.decode('utf-8')
        try:
            out = self.conn.query(query).as_dict()
        except:
            return 'Error3'

        result_set = [tuple(sorted(i.values())) for i in out]
        # result_set = [tuple(sorted(i.values(), key=lambda x:str(x))) for i in out]
        return result_set