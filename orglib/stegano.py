'''
自作ライブラリ第一弾
ステガノグラフィー

できること
・文字列を画像に埋め込む
・画像から埋め込まれた文字列を取り出す

原理

埋め込むとき
・前処理
与えられた文字列をutf-8に変換する
画像をrgbで読み込む
bの値を8で割った余りが1になるピクセルを抽出
bの値を1つ分だけ下げる
・本処理
bをベクトル化(一次元にする)←保留
16ピクセルごとに区切る(余りは無視)
変換した文字列のコードの値と同じ位置のピクセルの余りが1になるように操作
255の場合は7減らす
変換した値で画像再生成

読み込むとき
・本処理
画像をrgbで読み込む
bをベクトル化←保留
16ピクセルごとに区切る
bの値を8で割った余りが1になるピクセルを抽出
utf-8に対応させて復元
'''

from PIL import Image


# 書き込み
def stgWrite(path, text, savePath=None):
	'''
	param path: 文字列を埋め込む画像のパス．オプションを指定しなければ上書きする
	param text: 埋め込みたい文字列
	param savePath: オプション．指定するとそこに生成した画像を保存
	return: 処理の成否(成功でTrue)
	'''
	#オプション確認
	if savePath == None:
		savePath = path # パス指定されてなければpathと同じにする

	# 文字列をバイナリ化と16進数変換
	textData = text.encode("utf-8").hex()

	# print(len(textData))
	# print(type(textData))
	# print(textData)
	# print(bytes.fromhex(textData).decode())

	# 画像読み込み
	img = Image.open(path)

	# 文字列が画像に埋め込めるか計算
	if (img.width * img.height) // 16 < len(textData):
		print("This text is too long!!\n")
		return False

	# print(img.format, img.size, img.mode, img.info)
	# print(img.getpixel((0, 0)))

	r, g, b, a = img.split()

	# print(b.getpixel((0, 0)))

	# bの値の前処理
	_b = b.point((lambda val: val-1 if val%8 == 1 else val))

	# _b.show()

	# 情報埋め込み
	# 速度は遅いが仕方ない
	textPos = 0
	for h in range(_b.height):
		for w in range(0, _b.width - (_b.width % 16), 16): # 16ステップ
			_w = w + int(textData[textPos], 16)
			val = _b.getpixel((_w, h))
			_b.putpixel((_w, h), val - (val % 8) + 1) # 余りを1にする
			textPos += 1
			if textPos >= len(textData):
				break
		else: # 多重ループでのbreak処理
			continue
		break

	# 画像生成と保存
	Image.merge("RGBA", (r, g, _b, a)).save(savePath)

	return True

# 読み込み
def stgRead(path):
	'''
	param path: 文字列を読みたい画像のパス
	return: 読み取った文字列
	'''
	# 画像読み込みとバンド分割
	_, _, b, _ = Image.open(path).split() # b以外要らないので破棄

	# データ読み込み
	data = []
	for h in range(b.height):
		for w in range(0, b.width - (b.width % 16), 16):
			for i in range(16):
				if b.getpixel((w + i, h)) % 8 == 1: # 余りが1なら
					data.append(format(i, "x")) # データ抽出
					break

	# print("".join(data))

	return bytes.fromhex("".join(data)).decode() # テキストデータに変換してreturn


if __name__ == "__main__":
	print("これはテスト用です．もしこのメッセージが表示されている場合はコードの見直しを\n")

	text = """
test of the text, 0.123456789, てすとですよ試験

改行チェック
	"""

	# stgWrite("./test/dia_result01.png", text, savePath="./test/dia_result01_stg.png")

	print(stgRead("../output/data_all_N2_300_result11_31.png"))