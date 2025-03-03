import xlwings as xw
import serial
import time

def createtable(file_path):
    # 打开Excel程序，默认设置：程序可见，只打开不新建工作薄，屏幕更新关闭
    # app = xw.App(visible=True, add_book=False)
    # app.display_alerts = False
    # app.screen_updating = True

    # 文件位置：filepath，打开test文档，然后保存，关闭，结束程序
    filepath = r'D:\LIU research\Experiments\Experiments_2023_01_08_serial_data\es.xls'
    # wb = app.books.open('1.xlsx') # refer to the activated one
    wb = xw.Book()
    wb.activate()

    # wb.close()
    # app.quit()
    sheet = wb.sheets('Sheet1')
    row0 = ["current1", "current2", "current3", 'Joystick dangle', 'joystick bangle',
            'Actual dangle', 'Actual bangle']
    sheet.range(1, 1).value = row0
    sheet.autofit()
    wb.save(file_path)
    # print(sheet.name)
    # app.quit()


def write_data(file_path):
    # write the data inside the new book created by above function,
    # the sequence is current1 current2 current3 x, y bending direction, bending angle
    # the serial is always on, otherwise error is proposed in program
    # write the data every time when the command of recording occurs
    # from the camera program
    #
    if serial.isOpen():
        print('serial is open\n')
        wb = xw.Book(file_path)
        sheet1 = wb.sheets('sheet1')
        # acquire the edited rectange area, the most right side and bottom side cell
        last_cell = sheet1.used_range.last_cell
        # max row number
        last_row = last_cell.row
        serial.flushInput()
        while True:
            try:
                # size = serial.inWaiting()
                # print('size', size)
                # if size != 0:
                    response = serial.read(90).decode('utf-8')  # read the data
                    flag = False
                    data = ''
                    for i in range(len(response)):
                        if response[i] == '\n':
                            flag = True
                            continue
                        if response[i] == '\r':
                            break
                        if flag:
                            data = data + response[i]
                    print(data)
                    # decode the data and remove \r\n mark and seperate the str with comma
                    # s = data.decode('utf-8').rstrip('\r\n').split('\t')
                    s = data.split('\t')
                    print('s', s)
                    if len(s) != 4:
                        serial.flushInput()
                        continue
                    else:
                        try:
                            # fill the cell with list data as table
                            sheet1.range(last_row + 1, 1).options(expand='table').value = s
                            print(s)
                            serial.flushInput()  # clear the buffer
                            last_row = last_row + 1
                            time.sleep(0.5)
                            # break
                        except ValueError:
                            serial.flushInput()
                            continue
            except KeyboardInterrupt:
                wb.save()
                serial.close()
    else:
        print('error\n')


if __name__ == '__main__':
    file_path = '1.xlsx'
    # createtable(file_path)
    serial = serial.Serial('COM5', 115200, timeout=3)
    write_data(file_path)