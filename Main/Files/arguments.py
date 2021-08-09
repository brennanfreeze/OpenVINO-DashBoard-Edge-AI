import argparse
def get_args():
    parser = argparse.ArgumentParser("Run inference on an input video")
    #List of commands available to use in the command prompt
    v_desc = "Command to interpret which type of video to make an inference on. Note: default is camera but it can be converted to read a .mp4 by linking a video from file."
    d_desc = "Command to use device type. List includes: [CPU, VPU, GPU]."
    #Creates two argument types, one is required and one is optional to use
    parser._action_groups.pop()
    optional = parser.add_argument_group('OPTIONAL')
    optional.add_argument("-v", help = v_desc, default = 0)
    optional.add_argument("-d", help = d_desc, default = "CPU")

    #appoints arguments to be added when the command line call for "app.py" is called
    args = parser.parse_args()
    return args