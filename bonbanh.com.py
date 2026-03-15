from bs4 import BeautifulSoup
import requests
import time
import csv


with open('dataset.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Tên xe', 'Thương hiệu', 'Model', 'Năm sản xuất', 'Tình trạng', 'Số Km đã đi', 'Xuất xứ', 'Kiểu dáng', 'Hộp số', 'Động cơ', 'Số chỗ ngồi', 'Số cửa', 'Dẫn động', 'Giá thành'])


def write_csv(list):
    with open('dataset.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(list)


def find_brand_model(soup):
    global brands_and_models
    brands_and_models = {}
    soup_bm = soup.find_all('li', class_ = "menuparent")
    BaM = soup_bm[1:28]
    for brand in BaM:
        labels = [tag.text for tag in brand.find_all(['a', 'span'])]
        brands_and_models[labels[0]] = labels[1:]
    with open('Other brands and models.txt', 'r', encoding='utf-8') as txt:
        html_doc = txt.read()
    OtherBaMs = BeautifulSoup(html_doc, 'html.parser')
    others = OtherBaMs.find_all('li', class_='menuparent')
    others = others[1:]
    for i in others:
        other_labels = [other_tag.text for other_tag in i.find_all(['a', 'span'])]
        brands_and_models[other_labels[0]] = other_labels[1:]


def find_car(n):
    if n >= 2:
        address = "https://bonbanh.com/oto/page," + str(n)
    else:
        address = "https://bonbanh.com/oto"
    response = requests.get(address)
    response.encoding = 'utf-8'
    html_text = response.text
    soup = BeautifulSoup(html_text, 'lxml')
    if n == 1:
        global pages
        pages = soup.find('div', class_ = 'cpage').text
        pages = pages.split(" ")
        pages = pages[3].split(",")
        pages = int("".join(pages))
        find_brand_model(soup)
    cars_1 = soup.find_all('li', class_ = 'car-item row1')
    cars_2 = soup.find_all('li', class_ = 'car-item row2')
    cars = [cars_1, cars_2]

    for xe in cars:
        for car in xe:

            attributes = []

            ten_xe = car.find('div', class_ = 'cb2_02').text
            gia_tien = car.find('div', class_ = 'cb3').text

            #print (f'Ten xe:{ten_xe}')
            attributes.append(ten_xe)

            for brand in list(brands_and_models.keys()):
                if brand in ten_xe:
                    #print (f'Thuong hieu: {brand}')
                    attributes.append(brand)
                    for model in brands_and_models[brand]:
                        if model in ten_xe:
                            #print (f'Model: {model}')
                            attributes.append(model)
                            break
                    else:
                        #print ('Model: Khác')
                        attributes.append("Khác")
                    break
            else:
                #print ('Thuong hieu: Không xác định')
                attributes.append("Không xác định")
                #print ('Model: Không xác định')
                attributes.append("Không xác định")

            more_info = car.find('a').get('href')
            more_info = "https://bonbanh.com/" + more_info

            attributes.extend(get_detail(more_info))

            #print (f'Gia thanh: {gia_tien} \n')
            attributes.append(gia_tien)

            #print (attributes)
            #print ()
            write_csv(attributes)


def get_detail(address):
    response_inner = requests.get(address)
    response_inner.encoding = 'utf-8'
    html_text_inner = response_inner.text
    soup_inner = BeautifulSoup(html_text_inner, 'lxml')
    thong_so = soup_inner.find_all('div', {'class': 'row'})

    nam_san_xuat = thong_so[0].find('span', {'class': 'inp'}).text.strip()
    tinh_trang = thong_so[1].find('span', {'class': 'inp'}).text.strip()

    if tinh_trang == "Xe mới":
        so_Km_da_di = 0
        n = 2
    else:
        so_Km_da_di = thong_so[2].find('span', {'class': 'inp'}).text.strip()
        n = 3

    xuat_xu = thong_so[n].find('span', {'class': 'inp'}).text.strip()
    kieu_dang = thong_so[n+1].find('span', {'class': 'inp'}).text.strip()
    hop_so = soup_inner.find_all('div', {'class': 'row_last'})[0].find('span', {'class': 'inp'}).text.strip()
    dong_co = thong_so[n+2].find('span', {'class': 'inp'}).text.strip()
    so_cho_ngoi = thong_so[n+5].find('span', {'class': 'inp'}).text.strip()
    so_cua = thong_so[n+6].find('span', {'class': 'inp'}).text.strip()
    dan_dong = soup_inner.find_all('div', {'class': 'row_last'})[1].find('span', {'class': 'inp'}).text.strip()

    #print (f'Nam san xuat: {nam_san_xuat}')
    #print (f'Tinh trang: {tinh_trang}')
    #print (f'So Km da di: {so_Km_da_di}')
    #print (f'Xuat xu: {xuat_xu}')
    #print (f'Kieu dang: {kieu_dang}')
    #print (f'Hop so: {hop_so}')
    #print (f'Dong co: {dong_co}')
    #print (f'So cho ngoi: {so_cho_ngoi}')
    #print (f'So cua: {so_cua}')
    #print (f'Dan dong: {dan_dong}')

    return nam_san_xuat, tinh_trang, so_Km_da_di, xuat_xu, kieu_dang, hop_so, dong_co, so_cho_ngoi, so_cua, dan_dong
    

while True:
    find_car(1)
    print ('Waiting 5 seconds...')
    time.sleep(5)

    for i in range (2, pages):
        find_car(i)
        print ('Waiting 5 seconds...')
        time.sleep(5)