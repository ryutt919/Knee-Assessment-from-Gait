"""
한글 폰트 설정을 위한 스크립트
노트북에서 import하여 사용
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

def setup_korean_font():
    """
    시스템에 맞는 한글 폰트 자동 설정
    """
    system = platform.system()
    
    if system == 'Windows':
        # Windows에서 사용 가능한 한글 폰트 우선순위
        font_candidates = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 'AppleGothic']
    elif system == 'Darwin':  # macOS
        font_candidates = ['AppleGothic', 'NanumGothic', 'Arial Unicode MS']
    else:  # Linux
        font_candidates = ['NanumGothic', 'NanumBarunGothic', 'DejaVu Sans']
    
    # 사용 가능한 폰트 찾기
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    selected_font = None
    for font in font_candidates:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        plt.rcParams['font.family'] = selected_font
        plt.rcParams['axes.unicode_minus'] = False
        print(f"✅ 한글 폰트 설정 완료: {selected_font}")
        
        # 폰트 캐시 삭제 (필요시)
        try:
            import shutil
            cache_dir = fm.get_cachedir()
            cache_file = cache_dir + '/fontlist-v330.json'
            import os
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"   폰트 캐시 삭제: {cache_file}")
        except:
            pass
            
        return selected_font
    else:
        print("⚠️ 한글 폰트를 찾을 수 없습니다. 기본 폰트 사용")
        return None

if __name__ == "__main__":
    setup_korean_font()
    
    # 테스트
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, '한글 폰트 테스트\n가나다라마바사', 
            ha='center', va='center', fontsize=20)
    ax.set_title('한글 제목 테스트')
    ax.set_xlabel('X축 레이블')
    ax.set_ylabel('Y축 레이블')
    plt.tight_layout()
    plt.savefig('font_test.png', dpi=100)
    print("✅ 테스트 이미지 저장: font_test.png")
    plt.close()
