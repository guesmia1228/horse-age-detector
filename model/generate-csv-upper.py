import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    column_name = ['age', 'u1', 'u2', 'u3', 'u4', 'u5', 'u6',
                   'us1', 'us2', 'us3', 'us4', 'us5', 'us6', 'uss1', 'uss2', 'uss3', 'uss4', 'uss5', 'uss6']

    for xml_file in glob.glob(path + '/*.xml'):
        xml_type = None
        tree = ET.parse(xml_file)
        root = tree.getroot()
        xml_data = []
        filename = root.find('filename').text
        age = float(filename.split('-')[0])
        if age < 5 or age > 16:
            continue
        xml_data.append(age)
        visited_name = []
        for member in root.findall('object'):
            xmin = float(member[4][0].text)
            ymin = float(member[4][1].text)
            xmax = float(member[4][2].text)
            ymax = float(member[4][3].text)
            ratio = (ymax - ymin) / (xmax - xmin)
            visited_name.append(member[0].text)
            # value = (root.find('filename').text,
            #          int(root.find('size')[0].text),
            #          int(root.find('size')[1].text),
            #          member[0].text,
            #          int(member[4][0].text),
            #          int(member[4][1].text),
            #          int(member[4][2].text),
            #          int(member[4][3].text)
            #          )
            xml_data.append(str(ratio))
        if len(xml_data) == 19:
            xml_list.append(xml_data)
        else:
            # for key in column_name[1:len(column_name)]:
            #     if key not in visited_name:
            #         keyIndex = column_name.index(key) + 1
            #         # constant = 1.0
            #         # if age < 15:
            #         #     constant = age / 15
            #         xml_data.insert(keyIndex, float('nan'))
            # if len(xml_data) == 19:
            #     xml_list.append(xml_data)
            # else:
            #     print("Error:", len(xml_data))
            #     print("D", filename)
            continue

    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    # image_path = os.path.join(os.getcwd(), 'upper-teeth/annotations')
    image_path = 'D:/2018_Work/AgeDetection/training_data/2019_0129/Labelled/Labelled/Upper Combined 1-28-19 - Finished'
    df = xml_to_csv(image_path)
    df.to_csv('upper-teeth/horse_upper.csv', index=None)
    print('Successfully converted xml to csv.')


main()
