import os 
import sys

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)


from evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = ''
    settings.network_path = '/home/ziga/trackers/Stark/lib/test/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.result_plot_path = '/home/ziga/trackers/Stark/lib/test/result_plots/'
    settings.results_path = '/home/ziga/trackers/Stark/lib/test/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = '/home/ziga/trackers/Stark/lib/test/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''
    settings.totb_path = '/home/ziga/trackers/Stark/data/totb/'

    return settings

