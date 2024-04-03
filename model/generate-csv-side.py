import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_csv(path):
    right_xml_list = []
    left_xml_list = []
    all_labels = ['u1', 'b1','u2','b2','ug1','ug2', 'u5', 'b5', 'u6', 'b6', 'ug5', 'ug6']
    left_labels = ['u1', 'b1','u2','b2','ug1','ug2']
    right_labels = ['u5', 'b5', 'u6', 'b6', 'ug5', 'ug6']
    for xml_file in glob.glob(path + '/*.xml'):
        xml_type = None
        tree = ET.parse(xml_file)
        root = tree.getroot()
        xml_data = []
        filename = root.find('filename').text
        age = float(filename.split('-')[0])
        xml_data.append(age)
        visited_name = []
        temp_dict = {}

        for member in root.findall('object'):
            if member[0].text is None or member[0].text not in all_labels:
                continue
            if member[0].text in left_labels:
                xml_type = 0
            else:
                xml_type = 1

            xmin = float(member[4][0].text)
            ymin = float(member[4][1].text)
            xmax = float(member[4][2].text)
            ymax = float(member[4][3].text)
            ratio = (ymax - ymin) / (xmax - xmin)

            visited_name.append(member[0].text)

            # xml_data.append(str(ratio))
            temp_dict[member[0].text] = str(ratio)

        if xml_type == 0:
            for key in left_labels:
                if key in visited_name:
                    xml_data.append(temp_dict[key])
                else:
                    # temp_value = float(age) / 15.0
                    xml_data.append(float('nan'))
            if len(xml_data) == 7:
                left_xml_list.append(xml_data)
        else:
            for key in right_labels:
                if key in visited_name:
                    xml_data.append(temp_dict[key])
                else:
                    # temp_value = float(age) / 15.0
                    xml_data.append(float('nan'))
            if len(xml_data) == 7:
                right_xml_list.append(xml_data)

    l_column_name = ['age', 'u1', 'b1', 'u2', 'b2', 'ug1', 'ug2']
    r_column_name = ['age', 'u5', 'b5', 'u6', 'b6', 'ug5', 'ug6']

    xml_df_l = pd.DataFrame(left_xml_list, columns=l_column_name)
    xml_df_r = pd.DataFrame(right_xml_list, columns=r_column_name)
    return xml_df_l, xml_df_r


def main():
    # image_path = os.path.join(os.getcwd(), 'side-teeth/annotations')
    image_path = 'D:/2018_Work/AgeDetection/training_data/2019_0129/Labelled/Labelled/Side View - 1-21-19 - Finished'
    l_df, r_df = xml_to_csv(image_path)
    l_df.to_csv('side-teeth/horse_side_l.csv', index=None)
    r_df.to_csv('side-teeth/horse_side_r.csv', index=None)
    print('Successfully converted xml to csv.')


main()