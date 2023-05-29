import os

def load_images(path_to_folder):
    images = [os.path.join(path_to_folder, '1.jpg'),
              os.path.join(path_to_folder, '2.jpg'),
              os.path.join(path_to_folder, '3.jpg'),
              os.path.join(path_to_folder, '4.jpg'),
              os.path.join(path_to_folder, '5.jpg'),
              os.path.join(path_to_folder, '6.jpg'),
              os.path.join(path_to_folder, '7.jpg'),
              os.path.join(path_to_folder, '8.jpg'),
              os.path.join(path_to_folder, '9.jpg'),
              os.path.join(path_to_folder, '10.jpg'),
              os.path.join(path_to_folder, '11.jpg'),
              os.path.join(path_to_folder, '12.jpg'),
              os.path.join(path_to_folder, '15.jpg'),
              os.path.join(path_to_folder, '16.jpg'),
              os.path.join(path_to_folder, '17.jpg'),
              os.path.join(path_to_folder, '18.jpg'),
              os.path.join(path_to_folder, '19.jpg'),
              os.path.join(path_to_folder, '20.jpg'),
              os.path.join(path_to_folder, '21.jpg'),
              os.path.join(path_to_folder, '22.jpg'),
              os.path.join(path_to_folder, '23.jpg'),
              os.path.join(path_to_folder, '24.jpg'),
              os.path.join(path_to_folder, '25.jpg'),
              os.path.join(path_to_folder, '26.jpg'),
              os.path.join(path_to_folder, '27.jpg'),
              os.path.join(path_to_folder, '28.jpg'),
              os.path.join(path_to_folder, '29.jpg'),
              os.path.join(path_to_folder, '30.jpg'),
              os.path.join(path_to_folder, '31.jpg'),
              os.path.join(path_to_folder, '32.jpg'),
              os.path.join(path_to_folder, '33.jpg'),
              os.path.join(path_to_folder, '34.jpg'),
              os.path.join(path_to_folder, '35.jpg'),
              os.path.join(path_to_folder, '36.jpg'),
              os.path.join(path_to_folder, '37.jpg'),
              os.path.join(path_to_folder, '38.jpg'),
              os.path.join(path_to_folder, '39.jpg'),
              os.path.join(path_to_folder, '40.jpg'),
              os.path.join(path_to_folder, '41.jpg'),
              os.path.join(path_to_folder, '42.jpg'),
              os.path.join(path_to_folder, '43.jpg'),
              os.path.join(path_to_folder, '44.jpg'),
              os.path.join(path_to_folder, '45.jpg'),
              os.path.join(path_to_folder, '46.jpg'),
              os.path.join(path_to_folder, '47.jpg'),
              os.path.join(path_to_folder, '48.jpg'),
              os.path.join(path_to_folder, '49.jpg'),
              os.path.join(path_to_folder, '50.jpg'),
              os.path.join(path_to_folder, '51.jpg'),
              os.path.join(path_to_folder, '52.jpg'),
              os.path.join(path_to_folder, '53.jpg'),
              os.path.join(path_to_folder, '54.jpg'),
              os.path.join(path_to_folder, '55.jpg'),
              os.path.join(path_to_folder, '56.jpg'),
              os.path.join(path_to_folder, '57.jpg'),
              os.path.join(path_to_folder, '58.jpg'),
              os.path.join(path_to_folder, '59.jpg'),
              os.path.join(path_to_folder, '60.jpg'),
              os.path.join(path_to_folder, '61.jpg'),
              os.path.join(path_to_folder, '62.jpg'),
              os.path.join(path_to_folder, '63.jpg'),
              os.path.join(path_to_folder, '64.jpg'),
              os.path.join(path_to_folder, '65.jpg'),
              os.path.join(path_to_folder, '66.jpg'),
              os.path.join(path_to_folder, '67.jpg'),
              os.path.join(path_to_folder, '68.jpg'),
              os.path.join(path_to_folder, '69.jpg'),
              os.path.join(path_to_folder, '70.jpg'),
              os.path.join(path_to_folder, '71.jpg'),
              os.path.join(path_to_folder, '72.jpg'),
              os.path.join(path_to_folder, '73.jpg'),
              os.path.join(path_to_folder, '74.jpg'),
              os.path.join(path_to_folder, '75.jpg'),
              os.path.join(path_to_folder, '76.jpg'),
              os.path.join(path_to_folder, '77.jpg'),
              os.path.join(path_to_folder, '78.jpg'),
              os.path.join(path_to_folder, '79.jpg'),
              os.path.join(path_to_folder, '80.jpg'),
              os.path.join(path_to_folder, '81.jpg'),
              os.path.join(path_to_folder, '82.jpg'),
              ]
    return images

def load_pupil_centers():
    centers = [[21, 12], # 1
               [22, 14], # 2
               [22, 16], # 3
               [20, 17], # 4
               [16, 17], # 5
               [16, 16], # 6
               [23, 23], # 7
               [21, 22], # 8
               [26, 24], # 9
               [21, 18], # 10
               [21, 23], # 11
               [18, 19], # 12
               [28, 16], # 15
               [23, 14], # 16
               [22, 16], # 17
               [20, 16], # 18
               [25, 19], # 19
               [21, 18], # 20
               [24, 17], # 21
               [20, 17], # 22
               [21, 19], # 23
               [18, 18], # 24
               [21, 17], # 25
               [21, 16], # 26
               [21, 18], # 27
               [22, 21], # 28
               [24, 18], # 29
               [21, 18], # 30
               [27, 18], # 31
               [20, 15], # 32
               [26, 15], # 33
               [24, 13], # 34
               [26, 18], # 35
               [22, 20], # 36
               [23, 20], # 37
               [22, 20], # 38
               [21, 17], # 39
               [20, 16], # 40
               [24, 20], # 41
               [18, 19], # 42
               [23, 18], # 43
               [21, 19], # 44
               [21, 16], # 45
               [21, 19], # 46
               [24, 15], # 47
               [23, 17], # 48
               [25, 18], # 49
               [23, 18], # 50
               [21, 17], # 51
               [16, 16], # 52
               [25, 19], # 53
               [22, 17], # 54
               [23, 18], # 55
               [21, 17], # 56
               [21, 17], # 57
               [18, 18], # 58
               [24, 17], # 59
               [20, 19], # 60
               [25, 19], # 61
               [22, 17], # 62
               [23, 19], # 63
               [20, 19], # 64
               [24, 18], # 65
               [22, 18], # 66
               [22, 18], # 67
               [21, 17], # 68
               [24, 17], # 69
               [21, 18], # 70
               [25, 17], # 71
               [21, 16], # 72
               [24, 14], # 73
               [21, 12], # 74
               [23, 15], # 75
               [21, 14], # 76
               [26, 16], # 77
               [22, 14], # 78
               [26, 18], # 79
               [20, 16], # 80
               [25, 16], # 81
               [20, 16], # 82
               ]
    return centers



def load_eye_centers():
    centers = [[21, 15], # 1
               [21, 15], # 2
               [21, 16], # 3
               [20, 17], # 4
               [18, 18], # 5
               [19, 19], # 6
               [23, 22], # 7
               [21, 21], # 8
               [22, 20], # 9
               [17, 17], # 10
               [25, 20], # 11
               [21, 17], # 12
               [27, 16], # 15
               [22, 14], # 16
               [21, 16], # 17
               [20, 18], # 18
               [25, 20], # 19
               [21, 19], # 20
               [23, 17], # 21
               [20, 17], # 22
               [22, 20], # 23
               [19, 19], # 24
               [20, 17], # 25
               [20, 18], # 26
               [20, 18], # 27
               [21, 21], # 28
               [23, 19], # 29
               [20, 19], # 30
               [27, 19], # 31
               [20, 16], # 32
               [25, 16], # 33
               [23, 14], # 34
               [24, 17], # 35
               [20, 19], # 36
               [21, 20], # 37
               [20, 20], # 38
               [20, 19], # 39
               [19, 18], # 40
               [23, 20], # 41
               [18, 20], # 42
               [22, 18], # 43
               [19, 20], # 44
               [20, 17], # 45
               [20, 20], # 46
               [24, 15], # 47
               [23, 18], # 48
               [24, 19], # 49
               [21, 18], # 50
               [21, 18], # 51
               [19, 19], # 52
               [24, 19], # 53
               [21, 18], # 54
               [23, 19], # 55
               [20, 17], # 56
               [20, 18], # 57
               [18, 19], # 58
               [24, 18], # 59
               [20, 20], # 60
               [24, 18], # 61
               [21, 18], # 62
               [23, 20], # 63
               [19, 20], # 64
               [24, 19], # 65
               [20, 18], # 66
               [22, 18], # 67
               [21, 17], # 68
               [23, 16], # 69
               [21, 18], # 70
               [23, 18], # 71
               [20, 17], # 72
               [23, 14], # 73
               [21, 13], # 74
               [22, 16], # 75
               [21, 15], # 76
               [26, 18], # 77
               [22, 16], # 78
               [24, 19], # 79
               [19, 17], # 80
               [24, 17], # 81
               [20, 17], # 82
               ]
    return centers