from abr_analyze.plotting import Draw3dData
from abr_analyze.paths import figures_dir
import matplotlib.pyplot as plt
"""
A simple example of plotting parameters saved to the database on a 3d ax object
"""
# the number of samples to interpolate our data to, set to None for no
# interpolation
interpolated_samples=100
# list our tests and their relevant save locations
db_name = 'abr_analyze'
test = 'examples/test_1'
baseline = 'examples/baseline_1'

# Instantiate our arm drawing module
draw_3d = Draw3dData(db_name=db_name,
        interpolated_samples=interpolated_samples)

plt.figure()
ax = plt.subplot(111, projection='3d')
draw_3d.plot(
        ax=ax,
        save_location='%s/session000/run000'%test,
        parameters='ee_xyz',
        label='test',
        c='r')
draw_3d.plot(
        ax=ax,
        save_location='%s/session000/run000'%baseline,
        parameters='ee_xyz',
        label='baseline')
plt.title('My 3D Plot')

save_loc='%s/examples/3d_plot.png'%(figures_dir)
plt.savefig(save_loc)
plt.show()
print('Saved to %s'%save_loc)
