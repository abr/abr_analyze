from abr_analyze.utils import Draw2dData
from abr_analyze.utils.paths import figures_dir
import matplotlib.pyplot as plt
"""
A simple example of plotting parameters saved to the database on a 2d ax object
"""
# the number of samples to interpolate our data to, set to None for no
# interpolation
interpolated_samples=100
# list our tests and their relevant save locations
db_name = 'abr_analyze'
test = 'my_test_group/test1'
baseline = 'my_test_group/baseline1'

# Instantiate our arm drawing module
draw_2d = Draw2dData(db_name=db_name,
        interpolated_samples=interpolated_samples)

plt.figure()
ax = plt.subplot(111)
#NOTE: parameters can be a list if you would like them on the same ax object
draw_2d.plot(
        ax=ax,
        save_location=['%s/session000/run000'%test,
                       '%s/session000/run000'%baseline],
        parameters='u_base')
plt.title('My 2D Plot')

save_loc='%s/examples/2d_plot.png'%(figures_dir)
plt.savefig(save_loc)
plt.show()
print('Saved to %s'%save_loc)
