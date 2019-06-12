import argparse
import json
import os.path

import numpy as np
import pandas as pd

from fuzzy import predict_py, predict_cpu, predict_gpu

DEFAULT_DIM = 21


def generate_c_data_file(fsets_table, a0, a, b, filename='fuzzy_bench_data.c'):
	with open(filename, 'w') as f:
		print('const unsigned N = %d;' % a.shape[1], file=f)
		print('const unsigned n = %d;' % (len(fsets_table)-1), file=f)
		for i, fset_class in enumerate(fsets_table):
			for j, fset in enumerate(fset_class):
				print('const float fset_%d_%d[] = {%s};'
					  % (i, j, ', '.join(map(str, fset))), file=f)
		print('const float* fsets_buf[] = {%s};'
			  % ', '.join('fset_%d_%d' % (i, j)
						  for i, fset_class in enumerate(fsets_table)
							for j, fset in enumerate(fset_class)), file=f)
		print('const float** fsets_table[] = {%s};'
			  % ', '.join('fsets_buf + %d' % sum(map(lambda x: x.shape[0], fsets_table[:i]))
						  for i, _ in enumerate(fsets_table)), file=f)
		print('const unsigned fsets_lens[] = {%s};'
			  % ', '.join(map(lambda x: str(x.shape[0]), fsets_table)), file=f)
		print('const unsigned fsets_dims[] = {%s};'
			  % ', '.join(map(lambda x: str(x.shape[1]), fsets_table)), file=f)
		print('const float a0_table_buf[][%d] = {\n%s\n};'
			  % (max(map(len, a0)), ',\n'.join('\t{%s}' % ', '.join(map(str, a0i)) for a0i in a0)), file=f)
		print('const float* a0[] = {%s};' % ', '.join('a0_table_buf[%d]' % i for i, _ in enumerate(a0)), file=f)
		print('const unsigned char a_table_buf[][%d] = {\n%s\n};'
			  % (a.shape[1], ',\n'.join('\t{%s}' % ', '.join(map(str, ai)) for ai in a)), file=f)
		print('const unsigned char* a[] = {%s};' % ', '.join('a_table_buf[%d]' % i for i, _ in enumerate(a)), file=f)
		print('const unsigned char b[] = {%s};' % ', '.join(map(str, b)), file=f)
	print("File '%s' created." % filename)


def measure_n_times(predict_func, *args, num_iterations=1, **kwargs):
	from timeit import default_timer as timer

	elapsed = 0
	for _ in range(num_iterations):
		start = timer()
		b0 = predict_func(*args, **kwargs)
		end = timer()
		elapsed += end - start
	return elapsed


def measure_with_bound_time(predict_func, *args, max_measure_time, **kwargs):
	from timeit import default_timer as timer

	elapsed = 0
	num_iterations = 0
	while True:
		num_iterations += 1
		start = timer()
		b0 = predict_func(*args, **kwargs)
		end = timer()
		elapsed += end - start
		if elapsed > max_measure_time:
			if num_iterations > 1:
				elapsed -= end - start
			break
	return elapsed, num_iterations


def measure_time(predict_func, *args, num_iterations=1, max_measure_time=None, **kwargs):
	from timeit import default_timer as timer

	# print('--- Measurement started ---')
	if max_measure_time:
		elapsed, num_iterations = measure_with_bound_time(predict_func, *args, **kwargs, max_measure_time=max_measure_time)
	else:
		elapsed = measure_n_times(predict_func, *args, **kwargs, num_iterations=num_iterations)

	print('Average elapsed time for %d iterations is %f.' % (num_iterations, elapsed / num_iterations))


def infer(predict_func, fsets_table, a0, a, b):
	# print('--- Infering started ---')
	b0 = predict_func(fsets_table, a0, a, b)
	print(b0)


def build_fsets_table(class_nums):
	def mirror_down(arr, threshold, floor=None):
		need_be_mirrored = arr > threshold
		arr[need_be_mirrored] = threshold - arr[need_be_mirrored]
		arr[arr < floor] = floor
		return arr

	def f(i, n, h):
		growing = np.linspace(0, (n-1)/i, n, dtype=np.float32) if i else np.ones(n)
		descending = (i/(n-1) - np.linspace(0, 1, n)) / (1 - i/(n-1)) + 1 if i < n-1 else np.ones(n)
		res = np.minimum(growing, descending)
		res[res < 0.0] = 0.0
		return res.astype(np.float32)
	
	fsets_table = []
	for class_num in class_nums:
		fsets_table.append(np.array([f(i, class_num, class_num) for i in range(class_num)]).copy())
	return fsets_table


def build_fsets_metas(classes, types):
	"""
	Returns fsets_metas = [Union[
		<some int attribute> :: (x0, <step coefficient>),
		<some other attribute> :: { <class index>: <index in the fsets_table> }
	]]
	"""
	fsets_metas = []
	for clss, tp in zip(classes, types):
		if tp == np.dtype('int'):
			highest_cls = max(clss)
			lowest_cls = min(clss)
			fsets_metas.append((lowest_cls, (highest_cls-lowest_cls) // (len(clss)-1)))
		else:
			fsets_metas.append({cls: i for i, cls in enumerate(clss)})
	return fsets_metas


def build_rule(type, fsets_meta, row):
	if np.dtype(type) == np.dtype('int'):
		x0, k = fsets_meta
		return [int((x - x0) // k) for x in row]
	else:
		fsets_dict = fsets_meta
		return [fsets_dict[x] for x in row]
			

def build_rules(fsets_metas, rdata):
	return np.array([build_rule(rdata[col].dtype, fsets_meta, rdata[col].values) for col, fsets_meta in zip(rdata.columns, fsets_metas)], dtype=np.uint8).copy()


def fit_step_to_order(n, base):
	step = base
	while step * base**2 < n:
		for x in range(2, max(2, step)+1):
			if (step * x) % base == 0:
				step *= x
				break
	return step


def infer_classes_from_series(column):
	if column.dtype == np.dtype('int'):
		# NOTE(sergey): Building classes for an integer attribute must account for
		# that values grid can looks like -1, 0, 10, 20 and so on.
		lowest, highest = column.min(), column.max()
		step = fit_step_to_order(highest-lowest, 10)
		return list(range(lowest // step * step, (highest+step-1)//step*step, step))
	else:
		return list(set(column.values))


def build_fuzzy_system(rdata, *, num_inputs=None, classes=None):
	num_inputs = num_inputs or len(rdata.columns)-1

	if classes is None:
		classes = list(infer_classes_from_series(rdata[col]) for col in rdata.columns)
	class_nums = list(map(len, classes))
	
	input_cols = list(rdata.columns)[:num_inputs]
	cols = input_cols + [rdata.columns[-1]]
	rdata = rdata.loc[:, cols]
	rdata.drop_duplicates(subset=input_cols, keep='first', inplace=True)
	classes = classes[:num_inputs] + classes[-1:]
	class_nums = class_nums[:num_inputs] + class_nums[-1:]

	fsets_table = build_fsets_table(class_nums)
	fsets_metas = build_fsets_metas(classes, rdata.dtypes)
	rules = build_rules(fsets_metas, rdata).copy()

	# fsets_table = [[indices] for columns]
	return fsets_table, rules[:-1], rules[-1]


method_dict = {'py': predict_py, 'cpu': predict_cpu, 'gpu': predict_gpu}

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('COMMAND', choices=['generate', 'measure', 'infer'])
	# needed only in the case when measure or infer command is passed.
	parser.add_argument('--method', choices=['py', 'cpu', 'gpu'], default='gpu')
	# build_fuzzy_system parameters.
	parser.add_argument('--num_inputs', '-n', type=int, default=0)
	parser.add_argument('--need_link_headers', action='store_true')
	parser.add_argument('--classes')
	parser.add_argument('URL')
	args = parser.parse_args()

	header_expected = need_link_headers = args.need_link_headers
	rdata = pd.read_csv(args.URL, sep=';', header='infer' if header_expected else None)
	
	if args.classes is None:
		classes = None
	elif os.path.isfile(args.classes):
		with open(args.classes) as f:
			classes = json.load(f)
	else:
		try:
			classes = json.loads(args.classes)
		except:
			classes = None

	if classes:
		if need_link_headers:
			classes = [(classes.get(col, infer_classes_from_series(rdata[col]))) for col in rdata]
		elif list(classes) != list(rdata.columns):
			exit(1)
	else:
		# NOTE(sergey): Actually we can inter classes for each raw-data column right here, 
		# but we'll delegate this work to the build_fuzzy_system function.
		pass

	fsets_table, a, b = build_fuzzy_system(rdata, num_inputs=args.num_inputs, classes=classes)
	index = -1
	a0 = [fsets[x] for fsets, x in zip(fsets_table, a[:, index])]

	method = args.method
	predict_func = method_dict[method]

	command = args.COMMAND
	if command == 'generate':
		generate_c_data_file(fsets_table, a0, a, b)
	elif command == 'measure':
		measure_time(predict_func, fsets_table, a0, a, b, max_measure_time=10)
	elif command == 'infer':
		infer(predict_func, fsets_table, a0, a, b)
	else:
		print('Error')