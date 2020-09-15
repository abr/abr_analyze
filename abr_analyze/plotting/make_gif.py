"""
A class for converting a folder of images into a gif
"""
import os
import subprocess

from abr_analyze.paths import cache_dir, figures_dir


class MakeGif:
    def prep_fig_cache(self):
        """
        clear the gif figure cache to avoid adding other figures to the final
        animation
        returns the location of the gif_fig_cache
        """
        # set up save location for figures
        fig_cache = "%s/gif_fig_cache" % (cache_dir)

        if not os.path.exists(fig_cache):
            os.makedirs(fig_cache)

        # delete old files if they exist in the figure cache. These are used to
        # create a gif and need to be deleted to avoid the issue where the
        # current test has fewer images than what is already in the cache, this
        # would lead to the old images being appended to the end of the gif
        files = [f for f in os.listdir("%s" % fig_cache) if f.endswith(".png")]
        for ii, f in enumerate(files):
            if ii == 0:
                print("Deleting old temporary figures for gif creation...")
            os.remove(os.path.join("%s" % fig_cache, f))

        return fig_cache

    def create(self, fig_loc, save_name, save_loc=None, delay=5, res=None):
        """
        Module that checks fig_loc location for png files and creates a gif

        PARAMETERS
        ----------
        fig_loc: string
            location where .png files are saved
            NOTE: it is recommended to use a %03d numbering system (or more if more
            figures are used) to have leading zeros, otherwise gif may not be in
            order
        save_loc: string
            location to save gif
        save_name: string
            name to use for gif
        delay: int
            changs the delay between images in the gif
        res: list of two integers, Optional (Default: [1200, 1200])
            the pixel resolution of the final gif
        """
        if save_loc is None:
            save_loc = figures_dir

        res = [1200, 2000] if res is None else res

        if not os.path.exists(save_loc):
            os.makedirs(save_loc)
        bashCommand = ("convert -delay %i -loop 0 -resize %ix%i %s/*.png %s/%s.gif") % (
            delay,
            res[0],
            res[1],
            fig_loc,
            save_loc,
            save_name,
        )
        # bashCommand = ("convert -delay %i -loop 0 -deconstruct -quantize"%delay
        #                + " transparent -layers optimize -resize %ix%i"%(res[0],res[1])
        #                + " %s/*.png %s/%s.gif"
        #                %(fig_loc, save_loc, save_name))
        print("Creating gif...")
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        process.communicate()
        print("Finished")
        print("Gif saved to %s/%s" % (save_loc, save_name))
