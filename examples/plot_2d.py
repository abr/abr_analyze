"""
A simple example of plotting parameters saved to the database on a 2d ax object
"""
import matplotlib.pyplot as plt
from download_examples_db import check_exists as examples_db

from abr_analyze.paths import figures_dir
from abr_analyze.plotting import Draw2dData

examples_db()
# the number of samples to draw from the interpolated data
# set to None for no interpolation
interpolated_samples = 100
# list our tests and their relevant save locations
db_name = "abr_analyze_examples"
test = "test_1"
baseline = "baseline_1"

# Instantiate our arm drawing module
draw_2d = Draw2dData(db_name=db_name, interpolated_samples=interpolated_samples)

plt.figure()
ax = plt.subplot(111)
draw_2d.plot(
    ax=ax, save_location="%s/session000/run000" % test, parameters="q", label="test"
)
draw_2d.plot(
    ax=ax,
    save_location="%s/session000/run000" % baseline,
    # NOTE: parameters can be a list
    parameters=["q"],
    label="baseline",
    linestyle="--",
)

plt.title("My 2D Plot")

save_loc = "%s/2d_plot.png" % (figures_dir)
plt.savefig(save_loc)
plt.show()
print("Saved to %s" % save_loc)
