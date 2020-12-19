import numpy as np
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

def load_data():
    x_test = np.load('./x_test.npy')
    y_test = np.load('./y_test.npy')
    y_hm = np.load('./y_uncertainty_heatmap.npy')
    pred = np.load('./y_pred.npy')
    return x_test, y_test, y_hm, pred

#################
track_index = [14, 17, 21, 40, 76, 77, 99, 119]
#################
def draw_map(idx, points, unchange, heatmap, pred):
    ################
    # 画线和画点偏移量
    manual_line_offset_x = -7
    manual_line_offset_y = 7
    ################
    # 地图偏移量
    ###  右移 +   左移 -
    manual_bias_x = 165
    ###  上移 +   下移 -
    manual_bias_y = -35
    ################
    # 只显示想要的部分
    filter = False
    center_point = -1
    ################
    # 如果有已经保存的配置信息，读取
    load_saved = True
    # 将当前配置写入配置信息
    freeze_parameter = not True
    ################
    no_uncertainly = False
    no_mark = False
    zoom = 4
    ################

    ## load offset parameters
    try:
        if load_saved:
            manual_bias_x, manual_bias_y, manual_line_offset_x, manual_line_offset_y = np.load('output/{}.params.npy'.format(idx))
    except:
        pass

    if filter:
        max_hm = np.argmax(heatmap)
        if center_point >= 0:
            max_hm = center_point
        choosen = range(max(0,max_hm-20), min(max_hm+20,len(heatmap)))
        choosen = [x for x  in choosen]
        points = [points[i] for i in choosen]
        unchange = [unchange[i] for i in choosen]
        heatmap = [heatmap[i] for i in choosen]

    xpts = [x[1] for x in points]
    ypts = [x[0] for x in points]
    max_x, min_x = max(xpts), min(xpts)
    max_y, min_y = max(ypts), min(ypts)
    drt_x = max_x - min_x
    drt_y = max_y - min_y
    drt = max(drt_x, drt_y)
    canvas_width = 1200 * zoom
    canvas_height = 1200 * zoom
    canvas_padding = 60 * zoom
    real_canvas_width = canvas_width - 2 * canvas_padding
    real_canvas_height = canvas_height - 2 * canvas_padding

    ## 把图标移到中间
    drt_x_offset = max(0, drt_y-drt_x) / 2
    drt_y_offset = max(0, drt_x-drt_y) / 2

    def mapping_points(x, y):
        d_x = x - min_x + drt_x_offset
        d_y = y - min_y + drt_y_offset
        m_x = (d_x / drt) * real_canvas_width + canvas_padding + manual_line_offset_x
        m_y = canvas_height - ((d_y / drt) * real_canvas_height + canvas_padding) + manual_line_offset_y
        return m_x, m_y

    color_idx = 0
    def get_color():
        if color_idx == 0: return (255,0,0)
        elif color_idx == 1: return (0,0,255)
        elif color_idx == 2: return (0,0,0)
        elif color_idx == 3: return (255,0,0)
        else: return [int(x) for x in np.random.rand((3))*255]

    def get_color_by_heat(heat):
        return (int(255*(heat)), 100, int(255*(1-heat)))

    base_color = get_color()

    max_heat = max(heatmap)

    img = Image.new('RGBA', (canvas_width, canvas_height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    ## draw map
    padding_x_bias = (canvas_padding * drt) / real_canvas_width
    padding_y_bias = (canvas_padding * drt) / real_canvas_height
    print('PADDING BIAS: ', padding_x_bias, padding_y_bias)

    map_p1x = min_x - padding_x_bias - drt_x_offset
    map_p1y = min_y - padding_y_bias - drt_y_offset
    map_p2x = min_x + drt + padding_x_bias - drt_x_offset
    map_p2y = min_y + drt + padding_y_bias - drt_y_offset
    map_p_str = "{},{};{},{}".format(map_p1y, map_p1x, map_p2y, map_p2x)
    map_i_str = "{},{}|{},{}".format(map_p1y, map_p1x, map_p2y, map_p2x)
    print(map_p_str)
    map_c_str = "{},{}".format((min_x + max_x)/2, (min_y+max_y)/2)

    # 请求带斜线的颜色的road map
    mark_color = (69, 51, 85) # #453355
    try:
        img_mask = Image.open('mask{}.png'.format(idx))
    except:
        url = 'https://apis.map.qq.com/ws/staticmap/v2/?key=TORBZ-AGC3R-Y7YWF-WUVMG-CY3D3-BNFSC&size=1000*1000&bounds={}&path=color:0x45335500|{}'.format(map_p_str, map_i_str)
        print(url)
        response = requests.get(url)
        img_mask = Image.open(BytesIO(response.content))
        img_mask.save('mask{}.png'.format(idx))
    min_mark_x, min_mark_y, max_mark_x, max_mark_y = 1000, 1000, 0, 0
    img_mask_o = img_mask
    img_mask = np.array(img_mask)
    for i in range(img_mask.shape[0]):
        for j in range(img_mask.shape[1]):
            if img_mask[i][j][0] == mark_color[0] and img_mask[i][j][1] == mark_color[1] and img_mask[i][j][2] == mark_color[2]:
                min_mark_x = min(min_mark_x, j)
                min_mark_y = min(min_mark_y, i)
                max_mark_x = max(max_mark_x, j)
                max_mark_y = max(max_mark_y, i)
            else:
                img_mask[i][j][0] = img_mask[i][j][1] = img_mask[i][j][2] = 233
    #img_mask = Image.fromarray(img_mask, 'RGB')
    #img_mask.show()
    #img_mask_o.show()
    print(min_mark_x, min_mark_y, max_mark_x, max_mark_y)

    ## 请求背景
    try:
        img_bg = Image.open('bg{}.png'.format(idx))
    except:
        url = 'https://apis.map.qq.com/ws/staticmap/v2/?key=TORBZ-AGC3R-Y7YWF-WUVMG-CY3D3-BNFSC&size=1000*1000&bounds={}&maptype=satellite'.format(map_p_str)
        print(url)
        response = requests.get(url)
        img_bg = Image .open(BytesIO(response.content))
        img_bg.save('bg{}.png'.format(idx))
    img_bg_np = np.array(img_bg)
    #Image.fromarray(img_bg_np, 'RGBA').show()

    min_mark_x += manual_bias_x
    max_mark_x += manual_bias_x
    min_mark_y += manual_bias_y
    max_mark_y += manual_bias_y
    img_bg_new = img_bg_np[min_mark_y:max_mark_y, min_mark_x:max_mark_x]
    img_bg_new = Image.fromarray(img_bg_new, 'RGBA')
    img_bg_new = img_bg_new.resize((canvas_width, canvas_height))
    #img_bg_new.show()

    draw = ImageDraw.Draw(img)

    # 画heat的圆
    if not no_uncertainly:
        for i in range(len(points)):
            this_pt = points[i]
            threshold_uncertainly = 0.3
            heat = (heatmap[i] / max_heat)
            if (heat >= threshold_uncertainly):
                mx, my = mapping_points(this_pt[1], this_pt[0])
                print('over threshold: ', heat)
                circle_radius = int(10 * (1 + 20*(heat - threshold_uncertainly))*zoom)
                draw.ellipse([mx - circle_radius, my - circle_radius, mx + circle_radius, my + circle_radius],
                             #fill=(int(255 * (1 - heat)), int(255 * (1 - heat)), int(255 * (1 - heat)), 128),
                             fill = (255, 255, 255, int((heat-threshold_uncertainly)*208+10)),
                             outline=(255,255,255,200), width=int(4*zoom))

    # 画线
    for i in range(len(points)-1):
        this_pt, next_pt = points[i], points[i+1]
        if not unchange[i]:
            print("change at ", i)
            color_idx = color_idx + 1
            base_color = get_color()
        heat = (heatmap[i] / max_heat)
        #base_color = (255, 255, 255)
        heat = 1
        line_color = (int(heat*base_color[0]),int(heat*base_color[1]),int(heat*base_color[2]))
        #line_color = get_color_by_heat(heat)
        draw.line((*mapping_points(this_pt[1], this_pt[0]), *mapping_points(next_pt[1], next_pt[0])), fill=line_color, width=int(8*zoom))

    # 画方式改变的圆和端点
    upcount = 0
    upm, dnm = 0,0
    for i in range(len(points)):
        this_pt = points[i]
        if not unchange[i] or i==0 or i==len(points)-1:
            point_color = (255, 0, 0)
            # 没有预测出
            if unchange[i] != pred[i]:
                point_color = (0, 0, 255)
            if i==0: # 端点
                point_color = (255, 255, 0)
            elif i==len(points)-1:
                point_color = (51, 204, 153)
            darker_color = (int(point_color[0]*0.5),int(point_color[1]*0.5),int(point_color[2]*0.5))
            mx, my = mapping_points(this_pt[1], this_pt[0])
            circle_radius = int(17*zoom)
            draw.ellipse([mx - circle_radius, my - circle_radius, mx + circle_radius, my + circle_radius], fill=point_color,
                         outline=darker_color, width=int(3*zoom))

            if no_mark:
                continue
            #### 标出ABCD...
            # 计算斜率
            another_pt = points[i+5 if i<5 else i-5]
            x1, y1 = mapping_points(this_pt[1], this_pt[0])
            x2, y2 = mapping_points(another_pt[1], another_pt[0])
            if y1 == y2: k=1
            else: k=(x1-x2)/(y1-y2)
            k = abs(k)
            if k < 1: ## 显示在下面
                if dnm % 2 == 0: xb, yb = 25, -25
                else: xb, yb = 25, -25
                xb, yb = xb*zoom, yb*zoom
                dnm = dnm+1
            else:  ## 显示在左右
                if upm % 2 == 0: yb, xb = -65, -15
                else: yb, xb = -65, -15
                xb, yb = xb * zoom, yb * zoom
                upm = upm + 1
            wd = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'[upcount]
            draw.text((x1+xb, y1+yb), wd, fill=(0,0,0), font=ImageFont.truetype('arial', size=int(40*zoom)))
            print('MARK: ', wd, [x1+xb, y1+yb], 'up' if k>1 else 'right')
            upcount = upcount+1

    #img.show()
    img_bg_new.paste(img, (0,0), img)
    img_bg_new.show()
    img_bg_new.save('output/{}{}.png'.format(idx, '__no_uncertain' if no_uncertainly else ''))

    with open('output/{}.png.txt'.format(idx), 'w') as file:
        file.write(
            'manual_x: {}\nmanual_y: {}\nline_offfset_x: {}\nline_offset_y: {}'.format(manual_bias_x, manual_bias_y,
                                                                                       manual_line_offset_x,
                                                                                       manual_line_offset_y))
    if freeze_parameter:
        parameters = np.array([manual_bias_x, manual_bias_y, manual_line_offset_x, manual_line_offset_y])
        np.save('output/{}.params.npy'.format(idx), parameters)


def draw(i):
    draw_map(i, x[i], [t[0] for t in y[i]], hm[i], pred[i])

if __name__ == '__main__':
    x, y, hm, pred = load_data()
    if track_index is int:
        draw(track_index)
    else:
        for i in track_index:
            draw(i)
    #for track_idx in range(x.shape[0]):
    #    draw_map(track_idx, x[track_idx])