import re

##
# @brief replace a string 
#
# @param s: the string
# @param d: the dict
#
# @return 
def replace_with_dict(s, d):
    pattern = re.compile(r'\b(' + '|'.join(d.keys()) + r')\b')
    return pattern.sub(lambda x: d[x.group()], s)

