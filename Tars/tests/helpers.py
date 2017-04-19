
def display_samples(samples, format='e'):
    for i in samples:
        print ('{0:.16%s}' % format).format(i)
    print '='*20
