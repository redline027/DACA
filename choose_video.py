import numpy as np

def choose_video(base_path):
    #process path to make sure it's uniform
    """
    if ispc():
        base_path = strrep(base_path, '\\', '/')

    if base_path(end) ~= '/':
        base_path(end+1) = '/'

    #list all sub-folders
    contents = dir(base_path)
    names = {}
    for k = 1:numel(contents):
        name = contents(k).name
        if isdir([base_path name]) && ~strcmp(name, '.') && ~strcmp(name, '..'):
            names{end+1} = name  ##ok
        end
    end

    video_path = []

    #no sub-folders found
    if isempty(names):
        return video_path

    #choice GUI
    choice = listdlg('ListString',names, 'Name','Choose video', 'SelectionMode','single')

    if isempty(choice):
        #user cancelled
        video_path = []
    else:
        video_path = [base_path names{choice} '/']
"""

    video = 'deer'

    return video
