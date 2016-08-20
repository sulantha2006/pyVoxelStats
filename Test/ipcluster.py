import ipyparallel as ipp

rc = ipp.Client(profile=str.encode('sgeov'))
par_view = rc.direct_view(targets='all')
number_of_engines = len(par_view)
print(str.encode('sgeov'))
print(number_of_engines)
