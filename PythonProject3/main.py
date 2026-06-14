from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import random
import time
import pyautogui
import yaml


def get_element_screen_pos(driver, element):
    """获取元素相对于屏幕的实际坐标"""
    location = element.location
    size = element.size
    browser_pos = driver.get_window_position()
    
    # 计算元素中心点的屏幕坐标
    x = location['x'] + browser_pos['x'] + size['width'] // 2
    y = location['y'] + browser_pos['y'] + size['height'] // 2 + 80
    return x, y

def human_drag_trajectory(start_x, start_y, end_x, end_y, duration=None):
    """模拟人类拖拽轨迹：变速+随机抖动"""
    if duration is None:
        duration = random.uniform(1.0, 2.0)
    
    steps = int(duration * 50)
    pyautogui.mouseDown(button='left')
    time.sleep(random.uniform(0.05, 0.15))
    
    for i in range(1, steps + 1):
        progress = i / steps
        ease_progress = 1 - (1 - progress) ** 3
        target_x = start_x + (end_x - start_x) * ease_progress
        target_y = start_y + (end_y - start_y) * ease_progress
        jitter_y = random.gauss(0, 1.5)
        current_x = target_x
        current_y = target_y + jitter_y
        pyautogui.moveTo(current_x, current_y)
        time.sleep(duration / steps)
    pyautogui.moveTo(end_x - random.randint(1, 3), end_y)
    time.sleep(random.uniform(0.05, 0.1))
    pyautogui.moveTo(end_x, end_y)
    time.sleep(random.uniform(0.05, 0.1))
    
    pyautogui.mouseUp(button='left')

def huakuai_improved(driver):
    """改进版滑块验证：动态坐标+人类轨迹"""
    try:
        slider_btn = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '#rectBottom'))
        )
        start_x, start_y = get_element_screen_pos(driver, slider_btn)
        slider_btn.click()
        time.sleep(random.uniform(0.3, 0.6))
        drag_distance = random.randint(280, 380)
        end_x = start_x + drag_distance
        end_y = start_y
        human_drag_trajectory(start_x, start_y, end_x, end_y)
        
        time.sleep(random.uniform(1.0, 2.0))
        print("滑块验证完成")
        
    except Exception as e:
        print(f"滑块验证出错: {e}")
        huakuai_fallback()

def huakuai_fallback():
    """降级方案：当无法定位元素时使用固定坐标"""
    pyautogui.moveTo(random.randint(494, 496), 791, 0.2)
    time.sleep(1)
    pyautogui.dragTo(random.randint(888, 890), 791, 1)
    time.sleep(1)
    pyautogui.click(random.randint(652, 667), random.randint(793, 795))
    time.sleep(1)
    pyautogui.moveTo(random.randint(494, 496), 791, 0.2)
    time.sleep(1)
    pyautogui.dragTo(random.randint(888, 890), 791, 1)


def handle_dropdown(driver, selector):
    """针对第1题下拉框：JS 注入强制选择法"""
    try:
        q_box = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", q_box)
        time.sleep(0.5)
        q_id = selector.replace('#div', 'q')
        js_code = f"""
        var selectEl = document.querySelector('{q_id}');
        if (selectEl) {{
            var options = selectEl.options;
            var validIndices = [];
            for (var i = 0; i < options.length; i++) {{
                if (options[i].value && options[i].value !== "0" && options[i].text.indexOf("请选择") === -1) {{
                    validIndices.push(i);
                }}
            }}
            if (validIndices.length > 0) {{
                var randomIndex = validIndices[Math.floor(Math.random() * validIndices.length)];
                selectEl.selectedIndex = randomIndex;
                // 触发原生 change 事件
                selectEl.dispatchEvent(new Event('change', {{ bubbles: true }}));
                // 如果页面使用了 Select2，还需要触发它的 change
                if (window.jQuery && jQuery(selectEl).data('select2')) {{
                    jQuery(selectEl).trigger('change');
                }}
                return options[randomIndex].text;
            }}
        }}
        return null;
        """
        result = driver.execute_script(js_code)
        
        if result:
            print(f" {selector} 通过 JS 强制选择成功: {result}")
        else:
            print(f"️ {selector} JS 注入失败，尝试最后的物理保底")
            trigger = q_box.find_element(By.CSS_SELECTOR, ".select2-selection, .ui-select")
            trigger.click()
            time.sleep(0.8)
            pyautogui.press('down')
            time.sleep(0.2)
            pyautogui.press('enter')
            
    except Exception as e:
        print(f" {selector} 下拉框所有方法均失效: {e}")


def handle_matrix(driver, selector):
    """处理矩阵题"""
    try:
        matrix_box = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
        )
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", matrix_box)
        rows = matrix_box.find_elements(By.CSS_SELECTOR, "tr[id^='drv'], tr[sectype='1']")
        for row in rows:
            btns = row.find_elements(By.CSS_SELECTOR, "a.jqradio, .ui-radio, td")
            choices = btns[1:] if len(btns) > 1 else btns
            if choices:
                target = random.choice(choices)
                driver.execute_script("arguments[0].click();", target)
                time.sleep(random.uniform(0.2, 0.4))
        print(f" {selector} 矩阵题处理完成")
    except Exception as e:
        print(f" {selector} 矩阵题处理失败: {e}")


def handle_sort_question(driver, selector):
    """排序题：打乱顺序全选"""
    try:
        q_box = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", q_box)
        time.sleep(0.5)

        items = q_box.find_elements(By.CSS_SELECTOR, "li, .ui-checkbox, .jqcheck")
        
        if items:
            indices = list(range(len(items)))
            random.shuffle(indices) 
            
            print(f"开始对 {selector} 进行排序，共 {len(items)} 项")
            for idx in indices:

                driver.execute_script("arguments[0].click();", items[idx])
                time.sleep(random.uniform(0.4, 0.7))
            print(f" {selector} 排序题（全选乱序）完成")
    except Exception as e:
        print(f" {selector} 排序题处理失败: {e}")


def handle_question(driver, selector, answer_type='radio', valid_ans_count=1):
    """处理普通单选多选"""
    try:
        q_box = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", q_box)

        css = ".ui-radio, .ui-checkbox, .jqradio, .jqcheck, li"
        ans = q_box.find_elements(By.CSS_SELECTOR, css)
        ans = [a for a in ans if a.is_displayed() and "其他" not in a.text and a.text.strip()]
        
        if ans:
            num = 1 if answer_type == 'radio' else random.randint(1, min(valid_ans_count, len(ans)))
            for target in random.sample(ans, num):
                driver.execute_script("arguments[0].click();", target)
                time.sleep(0.3)
            print(f" {selector} 处理成功")
    except Exception as e:
        print(f" {selector} 跳过: {e}")


def scroll_to_element(driver, element):
    driver.execute_script("arguments[0].scrollIntoView();", element)


def tiankong(driver, num):
    answers_pool = [
        "无", "好", "good", "不错", "还可以", "满意", "挺好",
        "暂无", "没有", "不清楚", "一般般", "还行吧", "挺好的",
        "非常满意", "比较满意", "一般", "有待改进", "希望更好",
        "谢谢", "感谢", "辛苦了", "非常好", "很棒", "优秀",
        "A", "B", "C", "D", "E", "F", "G",
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
        "是", "否", "对", "错", "有", "没有"
    ]
    try:
        input_element = driver.find_element(By.CSS_SELECTOR, f'#q{num}')
        if random.random() > 0.1:
            answer = random.choice(answers_pool)
            input_element.send_keys(answer)
            time.sleep(random.uniform(0.3, 0.8))
    except NoSuchElementException:
        print(f"未找到第 {num} 题的填空题输入框。")


def renzheng(driver):
    wait = WebDriverWait(driver, 10)
    try:
        bth = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,
                                                         '#layui-layer1 > div.layui-layer-btn.layui-layer-btn- > a.layui-layer-btn0')))
        bth.click()
        time.sleep(random.uniform(0.8, 1.5))
        rectBottom = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#rectBottom')))
        rectBottom.click()
        time.sleep(random.uniform(1.5, 2.5))
        # 使用改进版滑块验证
        huakuai_improved(driver)
    except TimeoutException:
        print("本次未出现认证界面，继续进行后续操作")


def gundong(driver, distance):
    js = "var q=document.documentElement.scrollTop=" + str(distance)
    driver.execute_script(js)
    time.sleep(1)


def load_config(config_path="survey_config.yaml"):
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def execute_survey_from_yaml(driver, config_data):
    """根据 YAML 配置数据自动循环答题"""
    questions = config_data.get('questions', [])

    for q in questions:
        q_id = q['id']
        q_type = q['type']
        q_desc = q.get('desc', q_id)
        selector = f"#{q_id}"

        print(f"正在处理 -> {q_desc} [类型: {q_type}]")

        # 根据 YAML 中定义的 type，分发给不同的处理函数
        if q_type == 'dropdown':
            handle_dropdown(driver, selector)

        elif q_type == 'radio':
            handle_question(driver, selector, answer_type='radio')

        elif q_type == 'checkbox':
            max_c = q.get('max_choices', 1)
            handle_question(driver, selector, answer_type='checkbox', valid_ans_count=max_c)

        elif q_type == 'matrix':
            handle_matrix(driver, selector)

        elif q_type == 'sort':
            handle_sort_question(driver, selector)

        elif q_type == 'text':
            num = q['input_id'].replace('q', '')
            tiankong(driver, num)

        else:
            print(f"⚠️ 未知的题目类型: {q_type}，跳过该题。")

        # 题目之间的随机间隔（符合人类操作习惯）
        time.sleep(random.uniform(0.5, 1.2))


def zonghe(times, config_path="survey_config.yaml"):
    # 加载 YAML 配置
    config = load_config(config_path)
    settings = config.get('settings', {})
    url_survey = settings.get('url', 'https://v.wjx.cn/vm/PyKgkYl.aspx#')
    submit_selector = settings.get('submit_selector', '#ctlNext')
    success_keyword = settings.get('success_keyword', '您的答卷已经提交')

    # User-Agent池，每次使用不同的标识
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    ]

    for i in range(0, times):
        
        # 使用 Edge 浏览器（Windows 自带，无需安装）
        option = EdgeOptions()
        option.add_experimental_option('excludeSwitches', ['enable-automation'])
        option.add_experimental_option('useAutomationExtension', False)
        option.add_argument("--disable-blink-features=AutomationControlled")
        option.add_argument('--disable-extensions')
        option.add_argument('--no-sandbox')
        option.add_argument('--disable-dev-shm-usage')
        option.add_argument("--disable-infobars")  # 禁用信息栏

        option.add_argument(f'user-agent={random.choice(user_agents)}')
        
        driver = webdriver.Edge(options=option)

        driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument',
                               {'source': 'Object.defineProperty(navigator, "webdriver", {get: () => undefined})'})
        width = random.randint(1000, 1400)
        height = random.randint(800, 1000)
        driver.set_window_size(width, height)

        time.sleep(random.uniform(2, 5))

        driver.get(url_survey)

        page_loaded = False
        for selector in ['#div1', '.ui-controlgroup', '[id^="q"]', '.field-ui']:
            try:
                WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                print(f"页面加载完成（检测到 {selector}）")
                page_loaded = True
                break
            except TimeoutException:
                continue
        
        if not page_loaded:
            print("警告：未检测到标准题目元素，尝试继续执行...")
        
        time.sleep(3)  # 额外等待JavaScript渲染
        
        # 调试：打印页面中所有可能的题目相关元素
        try:
            # 尝试查找常见的问卷星题目容器
            possible_selectors = [
                "div[id^='div'][class*='control']",
                "div.ui-controlgroup",
                "div.field-ui",
                "div[class*='question']",
            ]
            
            found_elements = []
            for selector in possible_selectors:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    found_elements.append(f"{selector}: 找到 {len(elements)} 个")
            
            if found_elements:
                print(f"检测到的题目容器: {found_elements}")
            else:
                print("未找到标准题目容器，尝试通用选择器...")
                # 打印所有包含 'radio' 或 'checkbox' 的元素
                radios = driver.find_elements(By.CSS_SELECTOR, '.ui-radio')
                checkboxes = driver.find_elements(By.CSS_SELECTOR, '.ui-checkbox')
                print(f"单选按钮: {len(radios)} 个, 多选按钮: {len(checkboxes)} 个")
        except Exception as e:
            print(f"调试信息获取失败: {e}")

        execute_survey_from_yaml(driver, config)

        try:
            WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, submit_selector)))
            driver.find_element(By.CSS_SELECTOR, submit_selector).click()
        except TimeoutException:
            print("未找到下一步按钮，可能页面加载异常。")

        renzheng(driver)

        try:
            success_xpath = f'//div[@id="divdsc" and contains(text(), "{success_keyword}")]'
            success_msg = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, success_xpath))
            )
            if success_msg:
                print('问卷提交成功！')
        except TimeoutException:
            print('未检测到提交成功提示，可能出现问题。')

        print(f'已经提交了{i + 1}次问卷')
        driver.quit()

        if i < times - 1:
            wait_time = random.randint(10, 30)
            print(f'等待 {wait_time} 秒后继续下一次提交...')
            time.sleep(wait_time)


if __name__ == "__main__":
    zonghe(1, "survey_config.yaml")