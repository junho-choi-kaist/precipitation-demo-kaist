import numpy as np
import scipy as scp
import pandas as pd
import torch
import math
import datetime
from copy import deepcopy
from tqdm import tqdm
from PIL import Image

from app_constants import *
from app_subfunctions import *
from denormalized_visualization import my_cmap_radar_kma, my_cmap_radar_kma_px, my_cmap_radar_kma_pred, denormalize_hsr_torch

import plotly.express as px 
from dash import Dash, html, dcc, dash_table, callback, Output, Input, State, ctx
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from plotly.graph_objs.scatter import Line
import dash_bootstrap_components as dbc

################## 필요 변수 ##################
# Theme
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
chosen_theme = 'bootstrap'
dbc_theme = theme_dict[chosen_theme][0]
item_bg = theme_dict[chosen_theme][1]
item_text = theme_dict[chosen_theme][2]
external_stylesheets = [dbc_theme]


# Images
icon_path = 'assets/icon.png'


# 한글 --> 영어 변수 치환 dictionary
window_dict = {
    "대":'large',
    "중":'mid',
    "소":'small',
              }

# 모델과 데이터셋
print(f'{datetime.datetime.now()}: Loading model & datasets...')
model = load_model()
print(f'{datetime.datetime.now()}: Model loaded...')
train_dataset, valid_dataset, train_raw_dataset, valid_raw_dataset = load_datasets()
print(f'{datetime.datetime.now()}: Datasets loaded...')
common_map, lats, lons = get_latlon()
print(f'{datetime.datetime.now()}: Raw latitude/longitude loaded...')
similarity_items = load_similarity_items()
print(f'{datetime.datetime.now()}: Similarities loaded loaded...')
n_radar = train_dataset.input_dim
n_lead = 6

# 시간 리스트
time_list = list(pd.date_range("00:00", "23:50", freq="10min").strftime('%H:%M').values)
slice_list = [f'HSR{i}' for i in range(1, n_radar+1)]+['시간', '일', '월', '위도', '경도']

# 현재 시간
current_datetime = datetime.datetime.now()
current_y = current_datetime.year
current_m = current_datetime.month
current_d = current_datetime.day
current_h = current_datetime.hour
current_M = current_datetime.minute
current_date = datetime.date(current_y,current_m,current_d)
current_time = f"{str(current_h).zfill(2)}:{str(math.floor(current_M/10)*10).zfill(2)}"

# 차트 색
cmap_radar_kma, bounds_vals, bounds_labels, norm = my_cmap_radar_kma()  
cmap_radar_kma_px, bounds_vals_px, bounds_labels_px, norm_px = my_cmap_radar_kma_px()  
cmap_radar_kma_pred, bounds_vals_pred, bounds_labels_pred, norm_pred = my_cmap_radar_kma_pred(model.n_classes)  

# 빈 차트
empty_fig = go.Figure(data=[go.Scatter(x=[], y=[])])
empty_fig.update_layout(paper_bgcolor=item_bg)

# 글로벌 변수
current_input_idx = None
current_vis_item = None
old_selected_date = None
old_selected_similar = None

# 차트 설정
config={
        'edits': {
            'shapePosition': True,
                },
        }

################## 웹사이트 ##################
# 웹사이트 시작. CSS 적용 또는 BOOTSTRAP 적용


app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# 선택지 레이아웃
selector_layout = html.Div([
                  html.Div(children=[html.H2(children='선택 날짜', style=header2_style),
                                     html.Div(children=[dcc.DatePickerSingle(id='date-selector',
                                                          min_date_allowed=datetime.date(view_yr_start, 1, 1),
                                                          date=current_date,
                                                          display_format='Y/M/D')],style=item_style),
                                     html.H2(children='선택 시간', style=header2_style),
                                     html.Div(children=[dcc.Dropdown(time_list, 
                                                  value=current_time,
                                                  id='time-selector',
                                                  clearable=False,
                                                  style={"width":"30%","color":"black"},
                                                 )],style=dropdown_style),
                                    ],style=dash_margin),
                  html.Div(children=[html.H2(children='예측 기간', style=header2_style),
                                     html.Div(children=[dcc.Dropdown([i for i in range(1, n_lead+1)], 
                                                  value=1,
                                                  id='lead-selector',
                                                  clearable=False,
                                                  style={"width":"30%","color":"black"},
                                                 )],style=dropdown_style),
                                    ],style=dash_margin),
                  ])
    

# 입력값 레이아웃
empty_input = dcc.Tab(label=f'산출 전', children=[dcc.Graph(figure=empty_fig, config=config)], style=header4_style,id=f'graph-input')

input_layout = html.Div([dbc.Row([dbc.Col(html.H2(children='입력데이터', style=header3_style)),
                                   dbc.Col(html.Div(html.Button('산출', id='compute-input', n_clicks=0),style={'textAlign':'right'}))],justify='between'),
                                   dcc.Loading(dcc.Tabs(children=[empty_input],id='print-input')),
                ],style=dash_margin)

# 선택구역 강수량 레이아웃
empty_input_rainfall = dcc.Tab(label=f'산출 전', children=[dcc.Graph(figure=empty_fig, config=config)], style=header4_style,id=f'graph-input-rainfall')
rainfall_layout = html.Div([html.Div([html.H2(children='선택구역 순간 강수량', style=header3_style),
                                       dcc.Loading(dcc.Tabs(children=[empty_input_rainfall], id='print-input-rainfall'))
                ])],style=dash_margin)

# 예측 결과 레이아웃
empty_pred = dcc.Tab(label=f'산출 전', children=[dcc.Graph(figure=empty_fig, config=config)],style=header4_style,id=f'graph-pred')
empty_subpred = dcc.Tab(label=f'산출 전', children=[dcc.Graph(figure=empty_fig, config=config)],style=header4_style,id=f'graph-subpred')

pred_layout = html.Div([dbc.Row([dbc.Col(html.H2(children='예측치', style=header3_style)),
                                   dbc.Col(html.Div(html.Button('산출', id='compute-output', n_clicks=0),style={'textAlign':'right'}))],justify='between'),
                                   dcc.Loading(dcc.Tabs(children=[empty_pred, empty_subpred],id='print-output')),
                ],style=dash_margin)

                            
# 유사 결과 레이아웃
empty_sim = dcc.Tab(label=f'산출 전', children=[dcc.Graph(figure=empty_fig, config=config)], style=header4_style)

similar_layout = html.Div([dbc.Row([dbc.Col(html.H2(children='유사 사례', style=header3_style)),
                                   dbc.Col(html.Div(html.Button('산출', id='compute-similar', n_clicks=0),style={'textAlign':'right'}))],justify='between'),
                                   dcc.Loading(dcc.Tabs(children=[empty_sim],id='print-similar')),
                ],style=dash_margin)


# 유사결과 강수량 레이아웃
empty_sim_rainfall = dcc.Tab(label=f'산출 전', children=[dcc.Graph(figure=empty_fig, config=config)], style=header4_style)
similar_rainfall_layout = html.Div([html.Div([
    html.H2(children='유사사례 순간 강수량', style=header3_style),
    dcc.Loading(dcc.Tabs(children=[empty_sim_rainfall], id='print-similar-rainfall'))
])],style=dash_margin)
                            

  
                            

print(f'{datetime.datetime.now()}: Loading application...')
# 웹사이트 레이아웃
app.layout = html.Div([
    dbc.Stack([
        html.H1(children='프로토타입 기반 유사 관측치 탐색', style=header1_style),
        dbc.Row(selector_layout),
        dbc.Row([dbc.Col(input_layout,width=4),
                   dbc.Col(rainfall_layout,width=4),
                   dbc.Col(pred_layout,width=4)],align="start",justify="evenly",),
        dbc.Row([dbc.Col(similar_layout)],align="start",justify="evenly",),
        dbc.Row([dbc.Col(similar_rainfall_layout)],align="start",justify="evenly",),        
        ])
])


print(f'{datetime.datetime.now()}: Application loaded.')

# 입력값 생성
@callback(
    Output(component_id='print-input', component_property='children'),
    Input(component_id='compute-input', component_property='n_clicks'),
    Input(component_id='print-input', component_property='relayoutData'),
    State(component_id='date-selector', component_property='date'),
    State(component_id='time-selector', component_property='value'),
    State(component_id='lead-selector', component_property='value'),
    State(component_id='print-input', component_property='figure'),        
)

def update_input_graph(
                     n_clicks,
                     relayout_data,
                     selected_date, 
                     selected_time, 
                     target_lead, 
                     current_figure):
    tab_list = []
    if ((selected_date is not None) and (selected_time is not None) and 
        (target_lead is not None)):
        date_object = datetime.date.fromisoformat(selected_date)
        datestr = date_object.strftime('%Y%m%d')
        
        timestr = selected_time.replace(':','')
        target_datetime = datestr+timestr
                
        item = None
        i, item = find_date_in_dataset(target_datetime, 1, valid_dataset)            
        print(i, item)

        if item is not None:
            print(f'{datetime.datetime.now()}: Scaling data...')
            try:
                vis_item = valid_raw_dataset[i][0]
                print(f'{datetime.datetime.now()}: Raw data loaded...')
            except:
                vis_item = valid_dataset[i][0]
                vis_item[:n_radar+1] = denormalize_hsr(vis_item[:n_radar+1])
                print(f'{datetime.datetime.now()}: Proxied with normalized data...')

            vis_item_radar = vis_item[:n_radar]              

            print(f'{datetime.datetime.now()}: Making figure...')
            
            fnames = []
            for i in range(0,n_radar):
                fnames.append('temp/'+f'input_{datetime.datetime.now()}_{item[0]}_{i}.png')
            draw_geographic_graph(vis_item_radar, lats, lons, 
                                  cmap_radar_kma,
                                  bounds_labels, 
                                  bounds_vals,
                                  norm, fnames)                
            input_img_list = []
            for i in range(0,n_radar):
                fname = fnames[i]
                input_img_list.append(np.array(Image.open(fname)))
                os.remove(fname)
            input_img = np.stack(input_img_list)
                
            fig = px.imshow(input_img,
                            animation_frame=0,
                            height=single_window_size['height'],
                            width=single_window_size['width'],
                           )          
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
        
            fig.update_layout(margin=single_margin) 
            fig.update_layout(paper_bgcolor=item_bg)
            fig.update_layout(plot_bgcolor='white')

            slider_dates = []
            for d in item[2]:
                item_y = d[:4]
                item_m = d[4:6]
                item_d = d[6:8]
                item_h = d[8:10]
                item_min = d[10:]
                formatted_d = date_form.format(item_y, item_m, item_d, item_h, item_min)     
                slider_dates.append(formatted_d)            
            
            target_slider = make_custom_slider(slider_dates, fig)            
            fig.update_layout(sliders=target_slider)   
            fig.update_annotations(font_size=chart_font_size)            
            
        
            window_box = create_shape(single_window_limits['x0'],single_window_limits['y0'],(70,70))
            fig.update_layout(shapes=window_box)      
            tab = dcc.Tab(label=f'입력값', 
                  children=[dcc.Graph(figure=fig, config=config,id=f'graph-input')], style=header4_style)            
            tab_list.append(tab)            
            
            print(f'{datetime.datetime.now()}: Returning input figure...')

            return tab_list

        else:
            print(f'{datetime.datetime.now()}: Returning input empty...')
            tab = dcc.Tab(label=f'산출 전', 
                  children=[dcc.Graph(figure=empty_fig, config=config,id=f'graph-input')], style=header4_style)            
            tab_list.append(tab)            
            
            return tab_list            
    else:
        print(f'{datetime.datetime.now()}: Invalid input selection...')
        tab = dcc.Tab(label=f'산출 전', 
              children=[dcc.Graph(figure=empty_fig, config=config,id=f'graph-input')], style=header4_style)            
        tab_list.append(tab)            
            
        return tab_list    

# 입력값 생성
@callback(
    Output(component_id='graph-input', component_property='figure'),
    Input(component_id='graph-input', component_property='relayoutData'),
    State(component_id='graph-input', component_property='figure'),        
)    
    
def correct_window_size(relayout_data, current_figure):
    if ((relayout_data is not None) and 
        (current_figure['layout'].get('shapes') is not None)):
        current_shape = current_figure['layout'].get('shapes')[0]
        x0 = int(current_shape['x0'])
        x1 = int(current_shape['x1'])
        y0 = int(current_shape['y0'])
        y1 = int(current_shape['y1'])
        
        window_update = False
        if x0 < single_window_limits['x0']:
            x0 = single_window_limits['x0']
            window_update = True
        if y0 < single_window_limits['y0']:
            y0 = single_window_limits['y0']
            window_update = True
        if x1 > single_window_limits['x1']:
            x1 = single_window_limits['x1']
            window_update = True
        if y1 > single_window_limits['y1']:
            y1 = single_window_limits['y1']   
            window_update = True
        
        if window_update:
            window_shape = (x1-x0,y1-y0)
            window_box = create_shape(x0,y0,window_shape)
            current_figure['layout'].update(shapes=window_box)    
            print(f'{datetime.datetime.now()}: Correcting input window...')
        return current_figure   
    else:
        return current_figure    

# 입력값 중 선택구역 평균, 최소, 최대값 산출
@callback(
    Output(component_id='print-input-rainfall', component_property='children'),
    Input(component_id='graph-input', component_property='relayoutData'),    
    State(component_id='date-selector', component_property='date'),
    State(component_id='time-selector', component_property='value'),
    State(component_id='lead-selector', component_property='value'),
    State(component_id='graph-input', component_property='figure'),        
)

def update_input_rainfall(
                         relayout_data,
                         selected_date, 
                         selected_time, 
                         target_lead, 
                         current_figure):

    tab_list = []
    print(relayout_data)
    if ((selected_date is not None) and (selected_time is not None) and 
        (target_lead is not None) and (current_figure is not None)):
        date_object = datetime.date.fromisoformat(selected_date)
        datestr = date_object.strftime('%Y%m%d')
        
        timestr = selected_time.replace(':','')
        target_datetime = datestr+timestr
                
        item = None
        i, item = find_date_in_dataset(target_datetime, 1, valid_dataset)            

        if item is not None:
            print(f'{datetime.datetime.now()}: Scaling data...')
            try:
                vis_item = valid_raw_dataset[i][0]
                print(f'{datetime.datetime.now()}: Raw data loaded...')
            except:
                vis_item = valid_dataset[i][0]
                vis_item[:n_radar+1] = denormalize_hsr(vis_item[:n_radar+1])
                print(f'{datetime.datetime.now()}: Proxied with normalized data...')

            vis_item_radar = vis_item[:n_radar]              
            if current_figure['layout'].get('shapes') is not None:
                current_shape = current_figure['layout'].get('shapes')[0]
                bounding_box = shape_to_xy(current_shape, common_map, lons, lats)
            else:
                bounding_box = [(0,0),(vis_item.shape[1],vis_item.shape[2])]
            
            (x0,y0), (x1,y1) = bounding_box
            
            try:
                
                slider_dates = []
                for d in item[2]:
                    item_h = d[8:10]
                    item_min = d[10:]
                    formatted_d = f'{item_h}:{item_min}'    
                    slider_dates.append(formatted_d)         
                
                sub_input = vis_item_radar[:,x0:x1,y0:y1]
                sub_input_mean = torch.mean(sub_input,dim=[1,2])
                sub_input_max = torch.max(torch.max(sub_input,dim=1)[0],dim=1)[0]
                sub_input_min = torch.min(torch.min(sub_input,dim=1)[0],dim=1)[0]

                fig = make_subplots(rows=2, cols=1,
                                    subplot_titles=("평균", "최대", "최소")
                                   )
                fig.add_trace(
                        go.Scatter(x=slider_dates, y=sub_input_mean.numpy(), mode='lines'),
                        row=1, col=1
                    )
                fig.add_trace(
                        go.Scatter(x=slider_dates, y=sub_input_max.numpy(), mode='lines'),
                        row=2, col=1
                    )
                # fig.add_trace(
                #         go.Scatter(x=item[2], y=sub_input_min.numpy(), mode='lines'),
                #         row=3, col=1
                #     )       
                
             
                

                fig.update_layout(height=single_window_size['height'],
                                  width=single_window_size['width'])
                fig.update_layout(margin=chart_margin) 
                fig.update_layout(paper_bgcolor=item_bg)
                fig.update_layout(plot_bgcolor='white')
                fig.update_layout(showlegend=False) 
                
                xaxis, yaxis = make_axis_specs()
                fig.update_layout(xaxis=xaxis,
                                  yaxis=yaxis,
                                  xaxis2=xaxis,
                                  yaxis2=yaxis)

                fig.update_layout(
                    font_family="ui-san-serif",
                    font_color="black",
                    font_size=chart_font_size,
                    title_font_family="ui-san-serif",
                    title_font_color="black",
                    title_font_size=chart_font_size,
                )           
                
                fig.update_annotations(font_size=chart_font_size)        

                tab = dcc.Tab(label=f'선택구역 평균, 최대값', 
                              children=[dcc.Graph(figure=fig, config=config)], style=header4_style,
                              id=f'graph-input-rainfall')            
                tab_list.append(tab)      
            except:
                tab = dcc.Tab(label=f'산출 전',
                          children=[dcc.Graph(figure=empty_fig, config=config)], style=header4_style,
                          id=f'empty-input-rainfall')
                tab_list.append(tab)
            
            print(f'{datetime.datetime.now()}: Returning input rainfall figure...')

            return tab_list

        else:
            print(f'{datetime.datetime.now()}: Returning input empty...')
            tab = dcc.Tab(label=f'산출 전',
                          children=[dcc.Graph(figure=empty_fig, config=config)], style=header4_style,
                          id=f'empty-input-rainfall')
            tab_list.append(tab)            
            
            return tab_list            
    else:
        print(f'{datetime.datetime.now()}: Invalid input selection...')
        tab = dcc.Tab(label=f'산출 전',
                      children=[dcc.Graph(figure=empty_fig, config=config)], style=header4_style,
                      id=f'empty-input-rainfall')            
        tab_list.append(tab)            
            
        return tab_list        
    
    
    
# 예측치 생성
@callback(
    Output(component_id='print-output', component_property='children'),
    Input(component_id='compute-output', component_property='n_clicks'),
    State(component_id='date-selector', component_property='date'),
    State(component_id='time-selector', component_property='value'),
    State(component_id='lead-selector', component_property='value'),
    State(component_id='graph-input', component_property='figure'),
)
def update_pred_graph(n_clicks, selected_date, selected_time, target_lead, current_figure):
    tab_list = []
    if (selected_date is not None) and (selected_time is not None):
        date_object = datetime.date.fromisoformat(selected_date)
        datestr = date_object.strftime('%Y%m%d')
        
        timestr = selected_time.replace(':','')
        target_datetime = datestr+timestr
                
        item = None
        i, item = find_date_in_dataset(target_datetime, target_lead, valid_dataset)            
        print(i, item)
        
        if item is not None:
            print(f'{datetime.datetime.now()}: Computing prediction...')
            inputs = valid_dataset[i]
            pred = model(inputs[0].unsqueeze(0).to(device), torch.IntTensor([target_lead-1])).detach().cpu()
            pred = np.argmax(pred.squeeze(0).numpy(),axis=-1)
            
            print(f'{datetime.datetime.now()}: Computing pred figure...')    
            fname = 'temp/'+f'{datetime.datetime.now()}_pred_{item[0]}_{i}.png'
            draw_pred_graph(pred, lats, lons, 
                            graph_steps,
                                  cmap_radar_kma_pred,
                                  bounds_labels_pred, 
                                  bounds_vals_pred,
                                  norm_pred, fname)                
            input_img = np.array(Image.open(fname))
            os.remove(fname)
            
            item_y = item[0][:4]
            item_m = item[0][4:6]
            item_d = item[0][6:8]
            item_h = item[0][8:10]
            item_min = item[0][10:]
            formatted_date = date_form.format(item_y, item_m, item_d, item_h, item_min)
            
            fig = px.imshow(input_img,
                            title=f"{formatted_date}, 예측시간: {item[-1]} 시간 후",
                            height=single_window_size['height'],
                            width=single_window_size['width'],
                           )          
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)            
                        
            fig.update_layout(margin=pred_margin)
            fig.update_layout(paper_bgcolor=item_bg)
            fig.update_layout(
                title_font_family="ui-san-serif",
                title_font_color="black",
                title_font_size=chart_font_size,
            )              
            fig.update_annotations(font_size=chart_font_size)        
            
            
            tab = dcc.Tab(label=f'전체 예측치', 
                          children=[dcc.Graph(figure=fig, config=config)], style=header4_style,
                          id=f'graph-pred')
            
            tab_list.append(tab)
    
            # Input mask
            if current_figure['layout'].get('shapes') is not None:
                current_shape = current_figure['layout'].get('shapes')[0]
                fname = 'temp/'+f'{datetime.datetime.now()}_pred_sub_{item[0]}.png'
                
                bb = shape_to_xy(current_shape, common_map, lons, lats)
                pred_bb = np.flipud(pred)
                pred_bb = pred_bb[bb[0][0]:bb[1][0],bb[0][1]:bb[1][1]]
                pred_bb = np.flipud(pred_bb)
                lats_bb = lats[bb[0][0]:bb[1][0],bb[0][1]:bb[1][1]]
                lons_bb = lons[bb[0][0]:bb[1][0],bb[0][1]:bb[1][1]]                
                
                if np.prod(pred_bb.shape) > (1152/2)*(1440/2):
                    scale_factor = graph_steps
                else:
                    scale_factor = 1
                
                draw_pred_graph(pred_bb, 
                                lats_bb, 
                                lons_bb, 
                                scale_factor,
                                  cmap_radar_kma_pred,
                                  bounds_labels_pred, 
                                  bounds_vals_pred,
                                  norm_pred, fname)                
                input_img = np.array(Image.open(fname))
                os.remove(fname)

                fig = px.imshow(input_img,
                                title=f"{formatted_date}, 예측시간: {item[-1]} 시간 후",
                                height=single_window_size['height'],
                                width=single_window_size['width'],
                               )          
                fig.update_xaxes(showticklabels=False)
                fig.update_yaxes(showticklabels=False)            

                fig.update_layout(margin=pred_margin)
                fig.update_layout(paper_bgcolor=item_bg)
                fig.update_layout(
                    title_font_family="ui-san-serif",
                    title_font_color="black",
                    title_font_size=chart_font_size,
                )                
                fig.update_annotations(font_size=chart_font_size)        
                
                tab = dcc.Tab(label=f'예측치 중 선택구역', 
                  children=[dcc.Graph(figure=fig, config=config)], style=header4_style,id=f'graph-subpred')
                
                tab_list.append(tab)
                tab_list.reverse()
    
            print(f'{datetime.datetime.now()}: Returning pred figures...')
            return tab_list

        else:
            print(f'{datetime.datetime.now()}: Returning pred empty...')
            tab = dcc.Tab(label=f'산출 전', 
                children=[dcc.Graph(figure=empty_fig, config=config)], style=header4_style,id=f'graph-pred')              
            return [tab]
    else:
        print(f'{datetime.datetime.now()}: Invalid pred selection...')
        tab = dcc.Tab(label=f'산출 전', 
            children=[dcc.Graph(figure=empty_fig, config=config)], style=header4_style,id=f'graph-pred')         
        return [tab] 


# 유사치 생성
@callback(
    Output(component_id='print-similar', component_property='children'),
    Output(component_id='print-similar-rainfall', component_property='children'),
    Input(component_id='compute-similar', component_property='n_clicks'),
    State(component_id='date-selector', component_property='date'),
    State(component_id='time-selector', component_property='value'),
    State(component_id='lead-selector', component_property='value'),
    State(component_id='graph-input', component_property='figure'),
)
def update_similar_graph(n_clicks, selected_date, selected_time, dummy_lead,
                         current_figure):
    target_lead = 1
    
    if ((selected_date is not None) and (selected_time is not None) and
        (current_figure is not None)):
                
        date_object = datetime.date.fromisoformat(selected_date)
        datestr = date_object.strftime('%Y%m%d')
        
        timestr = selected_time.replace(':','')
        target_datetime = datestr+timestr
        current_shape = current_figure['layout'].get('shapes')[0]
        
        bounding_box = shape_to_xy(current_shape, common_map, lons, lats)
        (x0,y0),(x1,y1) = bounding_box
        
        item = None
        i, item = find_date_in_dataset(target_datetime, target_lead, valid_dataset)            
        print(i, item)

        tab_list = []
        tab_list2 = []
        if item is not None:
            prev_datetime = target_datetime
            print(f'{datetime.datetime.now()}: Computing hidden state...')
            input_data, gt_cls, gt, mask, lead, lead_cls = valid_dataset[i]
           
            model_args = {'lead':lead}

            x = compute_target_hidden_state(input_data, bounding_box, model, model_args)

            _, org_img_list, _, org_bb_list, date_list, inst_list = find_similar_input_idx(x, similarity_items, train_raw_dataset, topk=target_topk)

            input_list = []                
            for i in range(0,target_topk):
                fnames = []
                input_img_list = []
                for j in range(0,n_radar):
                    fnames.append('temp/' + f'{datetime.datetime.now()}_similar_{date_list[i]}_{j}.png')

                datestr = date_list[i]
                inst_datestr = inst_list[i]                    
                    
                (x0,y0), (x1,y1) = org_bb_list[i]
                
                sub_input = org_img_list[i][:n_radar,x0:x1,y0:y1]
                sub_input_mean = torch.mean(sub_input,dim=[1,2])
                sub_input_max = torch.max(torch.max(sub_input,dim=1)[0],dim=1)[0]
                sub_input_min = torch.min(torch.min(sub_input,dim=1)[0],dim=1)[0]
                
                item_y = datestr[:4]
                item_m = datestr[4:6]
                item_d = datestr[6:8]
                item_h = datestr[8:10]
                item_min = datestr[10:]
                formatted_date = date_form.format(item_y, item_m, item_d, item_h, item_min)                                      
                slider_dates = []
                for d in inst_datestr:
                    item_h = d[8:10]
                    item_min = d[10:]
                    formatted_d = f'{item_h}:{item_min}'    
                    slider_dates.append(formatted_d)                     

                fig1 = make_subplots(rows=2, cols=1,
                                    subplot_titles=("평균", "최대", "최소")
                                   )
                fig1.add_trace(
                        go.Scatter(x=slider_dates, y=sub_input_mean.numpy(), mode='lines'),
                        row=1, col=1
                    )
                fig1.add_trace(
                        go.Scatter(x=slider_dates, y=sub_input_max.numpy(), mode='lines'),
                        row=2, col=1
                    )
                # fig1.add_trace(
                #         go.Scatter(x=inst_list[i], y=sub_input_min.numpy(), mode='lines'),
                #         row=3, col=1
                #     )           
                
                fig1.update_layout(height=single_window_size['height'],
                                  width=single_window_size['width']) 
                fig1.update_layout(margin=chart_margin) 
                fig1.update_layout(paper_bgcolor=item_bg)
                fig1.update_layout(plot_bgcolor='white')
                fig1.update_layout(showlegend=False)   
                fig1.update_layout(title_text=formatted_date, height=single_window_size['height'])
                
                xaxis, yaxis = make_axis_specs()
                fig1.update_layout(xaxis=xaxis,
                                  yaxis=yaxis,
                                  xaxis2=xaxis,
                                  yaxis2=yaxis)                
                
                fig1.update_layout(
                    font_family="ui-san-serif",
                    font_color="black",
                    font_size=chart_font_size,
                    title_font_family="ui-san-serif",
                    title_font_color="black",
                    title_font_size=chart_font_size,
                )            
                
                fig1.update_annotations(font_size=chart_font_size)        
                
                tab_list2.append(dbc.Col(dcc.Graph(figure=fig1, config=config)))                

                draw_geographic_graph(org_img_list[i][:n_radar], lats, lons, 
                                      cmap_radar_kma,
                                      bounds_labels, 
                                      bounds_vals,
                                      norm, fnames)                          

                input_img_list = []
                for j in range(0,n_radar):
                    fname = fnames[j]
                    input_img_list.append(np.array(Image.open(fname)))
                    os.remove(fname)
                input_img = np.stack(input_img_list)    
                
          

                fig = px.imshow(input_img, 
                                animation_frame=0,
                                title=formatted_date,
                                height=single_window_size['height'],
                                width=single_window_size['width'],                                    
                               )                        

                x0, y0, x1, y1 = xy_to_shape(org_bb_list[i], current_shape, common_map, lons, lats)

                fig.add_shape(
                        dict(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, line_color="black")
                    )
                fig.update_xaxes(showticklabels=False)                
                fig.update_yaxes(showticklabels=False)
                
                fig.update_layout(
                    font_family="ui-san-serif",
                    font_color="black",
                    title_font_family="ui-san-serif",
                    title_font_color="black",
                    title_font_size=chart_font_size,
                )                   
                fig.update_annotations(font_size=chart_font_size)        

                slider_dates = []
                for d in inst_list[i]:
                    item_y = d[:4]
                    item_m = d[4:6]
                    item_d = d[6:8]
                    item_h = d[8:10]
                    item_min = d[10:]
                    formatted_d = date_form.format(item_y, item_m, item_d, item_h, item_min)     
                    slider_dates.append(formatted_d)
                
                target_slider = make_custom_slider(slider_dates, fig)            
                fig.update_layout(sliders=target_slider)   
                fig.update_layout(margin=single_margin) 
                fig.update_layout(paper_bgcolor=item_bg)   
                fig.update_layout(plot_bgcolor='white')
                fig.update_annotations(font_size=chart_font_size)        

                tab_list.append(dbc.Col(dcc.Graph(figure=fig, config=config)))

            print(f'{datetime.datetime.now()}: Returning similar figure...')
            return ([dcc.Tab(label=f'유사치', 
                      children=dbc.Row(tab_list), style=header4_style,id=f'graph-similar')], 
                    [dcc.Tab(label=f'유사치 평균, 최대값', 
                      children=dbc.Row(tab_list2), style=header4_style,id=f'graph-similar-rainfall')])
        else:
            print(f'{datetime.datetime.now()}: Returning similar empty...')
            tab = dcc.Tab(label=f'산출 전', 
                children=[dcc.Graph(figure=empty_fig, config=config)], style=header4_style,id=f'empty-similar')
            tab2 = dcc.Tab(label=f'산출 전', 
                children=[dcc.Graph(figure=empty_fig, config=config)], style=header4_style,id=f'empty-similar-rainfall')     
            return [tab], [tab2]    
    else:
        print(f'{datetime.datetime.now()}: Invalid similar selection...')
        tab = dcc.Tab(label=f'산출 전', 
            children=[dcc.Graph(figure=empty_fig, config=config)], style=header4_style,id=f'empty_similar')
        tab2 = dcc.Tab(label=f'산출 전', 
            children=[dcc.Graph(figure=empty_fig, config=config)], style=header4_style,id=f'empty-similar-rainfall')            
        return [tab], [tab2]    
    
    
    
# 웹사이트 실행 (VPN 필요)
if __name__ == '__main__':
    PORT = "8850"
    ADDRESS = "143.248.158.8"
    app.run(port=PORT,host=ADDRESS,debug=False)

################## notes ##################
# 1. 시작 후 조금 기다려야 Callback까지 로딩됨.

################## references ##################
# Multi-page app: https://community.plotly.com/t/how-to-pass-values-between-pages-in-dash/33739
# Annotation: https://dash.gallery/dash-image-annotation/
# Annotation 2: https://dash.plotly.com/annotations
# Annotation 3: https://github.com/plotly/dash-sample-apps/blob/main/apps/dash-image-segmentation/app.py
# Only shapes moving: https://github.com/plotly/plotly.js/blob/master/src/plot_api/plot_config.js#L51-L110
# https://community.rstudio.com/t/make-only-subset-of-shapes-editable-draggable-in-plotly/135552
# Editing shapes using callback: https://community.plotly.com/t/how-to-edit-delete-shapes-created-using-a-callback/69860/4
# Duplicate callback: https://dash.plotly.com/duplicate-callback-outputs
# Rows/cols: https://www.google.com/search?q=col-5&sca_esv=559003401&ei=MF3kZMHwJ6Cl0-kP-4W3yAU&ved=0ahUKEwjByN732O-AAxWg0jQHHfvCDVkQ4dUDCA8&uact=5&oq=col-5&gs_lp=Egxnd3Mtd2l6LXNlcnAiBWNvbC01MggQABiKBRiRAjIIEAAYigUYkQIyBxAAGIoFGEMyBxAAGIoFGEMyBxAAGIoFGEMyBxAAGIoFGEMyBRAAGIAEMgUQABiABDIFEAAYgAQyBRAAGIAESLkHUABYywZwAHgAkAEAmAGJAqAB1wWqAQUwLjMuMbgBA8gBAPgBAcICCxAuGIAEGMcBGNED4gMEGAAgQYgGAQ&sclient=gws-wiz-serp
# Rows/cols using dbc: https://dash-bootstrap-components.opensource.faculty.ai/docs/components/layout/
# Bootstrap themes: https://dash-bootstrap-components.opensource.faculty.ai/docs/themes/#available-themes
# Sharing data: https://dash.plotly.com/sharing-data-between-callbacks
# Colors: https://drafts.csswg.org/css-color/#color-syntax