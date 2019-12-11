import requests
import json
from tqdm import tqdm
import pandas as pd 

def get_info(json_data):
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36',
        'Authorization':'token 94073d48b209cbb1d4ad136533912e7862f77fe7'}
    data = {'Full Name':[],'Description':[], 'Stars':[],'Forks':[], 'Created Time':[], 'Size':[], 
            'Owner Type':[], 'Owner Location':[], 'Owner Followers':[], 'Repo_num':[], 'Company':[]}
    for info in json_data['items']:
        data['Full Name'].append(info['full_name'])
        data['Description'].append(info['description'])
        data['Stars'].append(info['stargazers_count'])
        data['Forks'].append(info['forks_count'])
        data['Created Time'].append(info['created_at'][:10])
        data['Size'].append(info['size'])
        data['Owner Type'].append(info['owner']['type'])
        usr_url = info['owner']['url']
        requests.adapters.DEFAULT_RETRIES = 10
        s = requests.session()
        s.keep_alive = False
        while(True):
            try:
                usr_r = requests.get(usr_url, headers = headers)
            except:
                print('%s二级页面访问失败,继续尝试访问' % info['full_name'])
                continue
            if usr_r.status_code == 200:
                data['Owner Location'].append(usr_r.json()['location'])
                data['Owner Followers'].append(usr_r.json()['followers'])
                data['Repo_num'].append(usr_r.json()['public_repos'])
                data['Company'].append(usr_r.json()['company'])
                break
    return pd.DataFrame(data)
    
if __name__ == '__main__':
    
    headers = {'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36',
        'Authorization':'token 85109d182be942e4f13cc76885d733115277bd71'}
    language = ['java', 'javascript', 'C', 'Cpp', 'Csharp']
    for item in language:
        data = pd.DataFrame(columns = ['Full Name', 'Description', 'Stars', 'Forks', 'Created Time', 'Size', 
            'Owner Type', 'Owner Location', 'Owner Followers', 'Repo_num', 'Company'])
        print('努力爬取%s数据中......' % item)
        for page in tqdm(range(1, 35)):
            #only first 1000 results are available
            url = 'https://api.github.com/search/repositories?q=language:%s&sort=stars&page=%s' % (item, str(page))
            while(True):
                try:
                    r = requests.get(url, headers = headers)
                except:
                    continue

                if r.status_code != 200:
                    print('页面%d访问失败' % page)
                else:
                    #访问成功不提示,通过进度条观察进度
                    data = pd.concat([data,get_info(r.json())], axis = 0, ignore_index = True)
                break

        data.to_excel('../data/Github_%s.xlsx' %  item)
        print('文件已保存至Github_%s.xlsx' % item)
