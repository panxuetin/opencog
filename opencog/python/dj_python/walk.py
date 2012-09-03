import os
for root, dirs, files in os.walk("./"):
    #print root
    for name in files:
        print "  " + name
        if name.find(".so"):
           print name 
    #for name in dirs:
        #print "  " + name
