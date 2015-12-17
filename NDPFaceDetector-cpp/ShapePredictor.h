#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <string>
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <Eigen/Geometry>
#include <cv.h>
#include <cstdint>

using namespace std;
//using namespace Eigen;
//using namespace cv;

template <typename T> inline bool serialize(const T& v, std::ostream& out){
	unsigned char buf[9], size = sizeof(T), neg;
	T item = v;
	if (item < 0){
		neg = 0x80; item *= -1;
	}
	else
		neg = 0;

	for (unsigned char i = 1; i <= sizeof(T); ++i){
		buf[i] = static_cast<unsigned char>(item & 0xFF);
		item >>= 8;
		if (item == 0) { size = i; break; }
	}

	std::streambuf* sbuf = out.rdbuf();
	buf[0] = size | neg;
	if (sbuf->sputn(reinterpret_cast<char*>(buf), size + 1) != size + 1){
		out.setstate(std::ios::eofbit | std::ios::badbit);
		return true;
	}
	return false;
}

template <typename T> inline bool deserialize(T& item, std::istream& in){
	unsigned char buf[8], size;
	bool is_negative;

	std::streambuf* sbuf = in.rdbuf();
	item = 0;
	int ch = sbuf->sbumpc();
	if (ch != EOF){
		size = static_cast<unsigned char>(ch);
	}
	else{
		in.setstate(std::ios::badbit);
		return true;
	}

	if (size & 0x80)
		is_negative = true;
	else
		is_negative = false;
	size &= 0x0F;
	// check if the serialized object is too big
	if (size > sizeof(T)) return true;
	if (sbuf->sgetn(reinterpret_cast<char*>(&buf), size) != size){
		in.setstate(std::ios::badbit);
		return true;
	}

	for (unsigned char i = size - 1; true; --i){
		item <<= 8;
		item |= buf[i];
		if (i == 0) break;
	}
	if (is_negative) item *= -1;
	return false;
}

inline void serialize(const float& item, std::ostream& out){
	int exp;
	const int digits = std::numeric_limits<float>::digits;
	long mantissa = static_cast<long>(frexp(item, &exp)*(((uint64)1) << digits));
	long exponent = exp - digits;
	serialize(mantissa, out);
	serialize(exponent, out);
}

inline void deserialize(float& item, std::istream& in){
	long mantissa; short exponent;
	deserialize(mantissa, in);
	deserialize(exponent, in);
	item = ldexp((float)mantissa, exponent);
}

inline void serialize(const cv::Mat& item, std::ostream& out){
	const unsigned long cols = static_cast<unsigned long>(item.cols);
	serialize(cols, out);
	const unsigned long rows = static_cast<unsigned long>(item.rows);
	serialize(rows, out);
	for (unsigned long i = 0; i < item.cols*item.rows; ++i)
		serialize(item.ptr<float>(0)[i], out);
}

inline void deserialize(cv::Mat& item, std::istream& in){
	long rows, cols;
	deserialize(cols, in);
	deserialize(rows, in);
	if (rows<0 || cols<0) { rows *= -1; cols *= -1; }

	item.create(rows, cols, CV_32F);
	for (unsigned long i = 0; i < rows*cols; ++i)
		deserialize(item.ptr<float>(0)[i], in);
}

inline void serialize(const Eigen::Vector2f& item, std::ostream& out){
	serialize(item(0), out);
	serialize(item(1), out);
}

inline void deserialize(Eigen::Vector2f& item, std::istream& in){
	deserialize(item(0), in);
	deserialize(item(1), in);
}

inline void serialize(const Eigen::Vector3i& item, std::ostream& out){
	serialize(item(0), out);
	serialize(item(1), out);
	serialize(item(2), out);
}

inline void deserialize(Eigen::Vector3i& item, std::istream& in){
	deserialize(item(0), in);
	deserialize(item(1), in);
	deserialize(item(2), in);
}

inline void serialize(const Eigen::Vector2i& item, std::ostream& out){
	serialize(item(0), out);
	serialize(item(1), out);
}

inline void deserialize(Eigen::Vector2i& item, std::istream& in){
	deserialize(item(0), in);
	deserialize(item(1), in);
}

template <typename T, typename alloc> void serialize(const std::vector<T, alloc>& item, std::ostream& out){
	const unsigned long size = static_cast<unsigned long>(item.size());
	serialize(size, out);
	for (unsigned long i = 0; i < item.size(); ++i)
		serialize(item[i], out);
}

template <typename T, typename alloc> void deserialize(std::vector<T, alloc>& item, std::istream& in){
	unsigned long size;
	deserialize(size, in);
	item.resize(size);
	for (unsigned long i = 0; i < size; ++i)
		deserialize(item[i], in);
}

class rand{
public:
	rand(){ init(); }
	rand(const std::string& seed_value){ init(); set_seed(seed_value); }
	virtual ~rand(){}
	void clear(){
		mt.seed();  seed.clear(); has_gaussian = false; next_gaussian = 0;
		// prime the generator a bit
		for (int i = 0; i < 10000; ++i) mt();
	}
	const std::string& get_seed(){ return seed; }
	void set_seed(const std::string& value){
		seed = value;
		// make sure we do the seeding so that using a seed of "" gives the same
		// state as calling this->clear()
		if (value.size() != 0){
			uint32_t s = 0;
			for (std::string::size_type i = 0; i < seed.size(); ++i){
				s = (s * 37) + static_cast<uint32_t>(seed[i]);
			}
			mt.seed(s);
		}
		else {
			mt.seed();
		}
		// prime the generator a bit
		for (int i = 0; i < 10000; ++i) mt();
		has_gaussian = false;
		next_gaussian = 0;
	}

	unsigned char get_random_8bit_number(){ return static_cast<unsigned char>(mt()); }
	uint16_t get_random_16bit_number(){ return static_cast<uint16_t>(mt()); }
	inline uint32_t get_random_32bit_number(){ return mt(); }
	inline uint64_t get_random_64bit_number(){
		const uint64_t a = get_random_32bit_number();
		const uint64_t b = get_random_32bit_number();
		return (a << 32) | b;
	}

	double get_random_double(){
		uint32_t temp;
		temp = rand::get_random_32bit_number();
		temp &= 0xFFFFFF;
		double val = static_cast<double>(temp);
		val *= 0x1000000;
		temp = rand::get_random_32bit_number();
		temp &= 0xFFFFFF;
		val += temp;
		val /= max_val;
		if (val < 1.0)
			return val;
		else
			return 1.0 - std::numeric_limits<double>::epsilon();
	}

	float get_random_float(){
		uint32_t temp;
		temp = rand::get_random_32bit_number();
		temp &= 0xFFFFFF;
		const float scale = 1.0 / 0x1000000;
		const float val = static_cast<float>(temp)*scale;
		if (val < 1.0f) return val;
		else
			return 1.0f - std::numeric_limits<float>::epsilon();
	}

	double get_random_gaussian(){
		if (has_gaussian){
			has_gaussian = false;
			return next_gaussian;
		}
		double x1, x2, w;
		const double rndmax = std::numeric_limits<uint32_t>::max();
		// Generate a pair of Gaussian random numbers using the Box-Muller transformation.
		do{
			const double rnd1 = get_random_32bit_number() / rndmax;
			const double rnd2 = get_random_32bit_number() / rndmax;
			x1 = 2.0 * rnd1 - 1.0;
			x2 = 2.0 * rnd2 - 1.0;
			w = x1 * x1 + x2 * x2;
		} while (w >= 1.0);

		w = std::sqrt((-2.0 * std::log(w)) / w);
		next_gaussian = x2 * w;
		has_gaussian = true;
		return x1 * w;
	}
private:
	void init(){
		// prime the generator a bit
		for (int i = 0; i < 10000; ++i) mt();
		max_val = 0xFFFFFF;
		max_val *= 0x1000000;
		max_val += 0xFFFFFF;
		max_val += 0.01;
		has_gaussian = false;
		next_gaussian = 0;
	}
	std::mt19937 mt;
	std::string seed;
	double max_val;
	bool has_gaussian;
	double next_gaussian;
};

struct split_feature{
	unsigned long idx1, idx2;
	float thresh;

	friend inline void serialize(const split_feature& item, std::ostream& out){
		serialize(item.idx1, out);
		serialize(item.idx2, out);
		serialize(item.thresh, out);
	}
	friend inline void deserialize(split_feature& item, std::istream& in){
		deserialize(item.idx1, in);
		deserialize(item.idx2, in);
		deserialize(item.thresh, in);
	}
};

inline unsigned long left_child(unsigned long idx)  { return 2 * idx + 1; }
inline unsigned long right_child(unsigned long idx) { return 2 * idx + 2; }

struct regression_tree{
	std::vector<split_feature> splits;
	std::vector<cv::Mat> leaf_values;
	inline const cv::Mat& operator()(const std::vector<float>& feature_pixel_values) const{
		unsigned long i = 0;
		while (i < splits.size()){
			if (feature_pixel_values[splits[i].idx1] - feature_pixel_values[splits[i].idx2] > splits[i].thresh)
				i = left_child(i);
			else
				i = right_child(i);
		}
		return leaf_values[i - splits.size()];
	}

	friend void serialize(const regression_tree& item, std::ostream& out){
		serialize(item.splits, out);
		serialize(item.leaf_values, out);
	}
	friend void deserialize(regression_tree& item, std::istream& in){
		deserialize(item.splits, in);
		deserialize(item.leaf_values, in);
	}
};

inline Eigen::Vector2f location(const cv::Mat& shape, unsigned long idx){
	return Eigen::Vector2f(shape.ptr<float>(0)[idx * 2], shape.ptr<float>(0)[idx * 2 + 1]);
}
inline float length_squared(Eigen::Vector2f p){ return  p(0)*p(0) + p(1)*p(1); }
inline float length_(Eigen::Vector2f p){ return  sqrtf(p(0)*p(0) + p(1)*p(1)); }
inline unsigned long nearest_shape_point(const cv::Mat& shape, const Eigen::Vector2f& pt){
	// find the nearest part of the shape to this pixel
	float best_dist = std::numeric_limits<float>::infinity();
	const unsigned long num_shape_parts = shape.cols / 2;
	unsigned long best_idx = 0;
	for (unsigned long j = 0; j < num_shape_parts; ++j){
		const float dist = length_squared(location(shape, j) - pt);
		if (dist < best_dist){
			best_dist = dist; best_idx = j;
		}
	}
	return best_idx;
}

inline void create_shape_relative_encoding(const cv::Mat& shape, const std::vector<Eigen::Vector2f>& pixel_coordinates,
	std::vector<unsigned long>& anchor_idx, std::vector<Eigen::Vector2f>& deltas){
	anchor_idx.resize(pixel_coordinates.size());
	deltas.resize(pixel_coordinates.size());

	for (unsigned long i = 0; i < pixel_coordinates.size(); ++i){
		anchor_idx[i] = nearest_shape_point(shape, pixel_coordinates[i]);
		deltas[i] = pixel_coordinates[i] - location(shape, anchor_idx[i]);
	}
}

Eigen::Matrix<float, 2, 3> find_similarity_transform(const std::vector<Eigen::Vector2f>& from_points, const std::vector<Eigen::Vector2f>& to_points){
	Eigen::Vector2f mean_from, mean_to;
	double sigma_from = 0, sigma_to = 0;
	Eigen::Matrix<double, 2, 2> cov; cov.setConstant(0);

	mean_from.setConstant(0); mean_to.setConstant(0);
	for (unsigned long i = 0; i < from_points.size(); ++i){
		mean_from += from_points[i];
		mean_to += to_points[i];
	}
	mean_from /= (float)from_points.size();
	mean_to /= (float)from_points.size();

	for (unsigned long i = 0; i < from_points.size(); ++i){
		sigma_from += length_squared(from_points[i] - mean_from);
		sigma_to += length_squared(to_points[i] - mean_to);
		cov += ((to_points[i] - mean_to)*(from_points[i] - mean_from).transpose()).cast<double>();
	}

	sigma_from /= from_points.size();
	sigma_to /= from_points.size();
	cov /= from_points.size();
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(cov, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::MatrixXd u = svd.matrixU();
	Eigen::MatrixXd v = svd.matrixV();
	Eigen::MatrixXd d = svd.singularValues();
	Eigen::Matrix<double, 2, 2> s; s.setIdentity();

	if (cov.determinant()< 0 || (cov.determinant() == 0 && u.determinant()*v.determinant()<0)){
		if (d(1) < d(0))
			s(1, 1) = -1.0f;
		else
			s(0, 0) = -1.0f;
	}

	Eigen::Matrix<double, 2, 2> r = u*s*v.transpose();
	float c = 1;
	if (sigma_from != 0){
		Eigen::Matrix<double, 2, 2> dd; dd.setIdentity();
		dd(0, 0) = d(0); dd(1, 1) = d(1);
		c = (float)(1.0 / sigma_from * (dd*s).trace());
	}
	Eigen::Vector2f t = mean_to - c*r.cast<float>()*mean_from;
	Eigen::Matrix<float, 2, 2> rot;
	rot = c*r.cast<float>();
	Eigen::Matrix<float, 2, 3> A;
	A(0, 0) = rot(0, 0); A(0, 1) = rot(0, 1); A(1, 0) = rot(1, 0); A(1, 1) = rot(1, 1);
	A(0, 2) = t(0); A(1, 2) = t(1);
	return A;
}

inline Eigen::Matrix<float, 2, 3> find_tform_between_shapes(const cv::Mat& from_shape, const cv::Mat& to_shape){
	std::vector<Eigen::Vector2f > from_points, to_points;
	const unsigned long num = from_shape.cols / 2;
	from_points.reserve(num);
	to_points.reserve(num);
	Eigen::Matrix<float, 2, 3> A;
	if (num == 1){// Just use an identity transform if there is only one landmark.
		A.setConstant(0); A(0, 0) = A(1, 1) = 1.0f;
		return A;
	}

	for (unsigned long i = 0; i < num; ++i){
		from_points.push_back(location(from_shape, i));
		to_points.push_back(location(to_shape, i));
	}
	return find_similarity_transform(from_points, to_points);
}

inline Eigen::Matrix<float, 2, 3> normalizing_tform(const cv::Rect& r){
	std::vector<Eigen::Vector2f > from_points, to_points;

	from_points.push_back(Eigen::Vector2f(r.x, r.y)); to_points.push_back(Eigen::Vector2f(0, 0));
	from_points.push_back(Eigen::Vector2f(r.x + r.width - 1, r.y)); to_points.push_back(Eigen::Vector2f(1, 0));
	from_points.push_back(Eigen::Vector2f(r.x + r.width - 1, r.y + r.height - 1)); to_points.push_back(Eigen::Vector2f(1, 1));
	return find_similarity_transform(from_points, to_points);
}

inline Eigen::Matrix<float, 2, 3> unnormalizing_tform(const cv::Rect& r){
	std::vector<Eigen::Vector2f > from_points, to_points;
	to_points.push_back(Eigen::Vector2f(r.x, r.y)); from_points.push_back(Eigen::Vector2f(0, 0));
	to_points.push_back(Eigen::Vector2f(r.x + r.width - 1, r.y)); from_points.push_back(Eigen::Vector2f(1, 0));
	to_points.push_back(Eigen::Vector2f(r.x + r.width - 1, r.y + r.height - 1)); from_points.push_back(Eigen::Vector2f(1, 1));
	return find_similarity_transform(from_points, to_points);
}

void extract_feature_pixel_values(const cv::Mat& img_, const cv::Rect& rect,
	const cv::Mat& current_shape, const cv::Mat& reference_shape,
	const std::vector<unsigned long>& reference_pixel_anchor_idx,
	const std::vector<Eigen::Vector2f >& reference_pixel_deltas,
	std::vector<float>& feature_pixel_values) {

	const Eigen::Matrix<float, 2, 3> tform_to_sp = find_tform_between_shapes(reference_shape, current_shape);
	Eigen::Matrix<float, 2, 2> tform;
	tform(0, 0) = tform_to_sp(0, 0); tform(0, 1) = tform_to_sp(0, 1);
	tform(1, 0) = tform_to_sp(1, 0); tform(1, 1) = tform_to_sp(1, 1);

	const Eigen::Matrix<float, 2, 3> tform_to_img = unnormalizing_tform(rect);

	feature_pixel_values.resize(reference_pixel_deltas.size());
	for (unsigned long i = 0; i < feature_pixel_values.size(); ++i){
		// Compute the point in the current shape corresponding to the i-th pixel and
		// then map it from the normalized shape space into pixel space.
		Eigen::Vector2f x = tform*reference_pixel_deltas[i] + location(current_shape, reference_pixel_anchor_idx[i]);
		Eigen::Vector2f p = tform_to_img * Eigen::Vector3f(x(0), x(1), 1);
		if (p(0) >= 0 && p(0)<img_.cols - 1 && p(1) >= 0 && p(1)<img_.rows - 1)
			feature_pixel_values[i] = img_.at<uchar>((int)(p(1) + 0.5), (int)(p(0) + 0.5));
		else
			feature_pixel_values[i] = 0;
	}
}

struct obj{
	cv::Rect r;
	std::vector<Eigen::Vector2f> parts;
};

class shape_predictor{
public:
	shape_predictor(){}
	shape_predictor(const cv::Mat& initial_shape_, const std::vector<std::vector<regression_tree> >& forests_,
		const std::vector<std::vector<Eigen::Vector2f > >& pixel_coordinates) : initial_shape(initial_shape_), forests(forests_){
		anchor_idx.resize(pixel_coordinates.size());
		deltas.resize(pixel_coordinates.size());
		// Each cascade uses a different set of pixels for its features.  We compute
		// their representations relative to the initial shape now and save it.
		for (unsigned long i = 0; i < pixel_coordinates.size(); ++i)
			create_shape_relative_encoding(initial_shape, pixel_coordinates[i], anchor_idx[i], deltas[i]);
	}
	unsigned long num_parts() const{ return initial_shape.cols / 2; }
	obj operator()(const cv::Mat& img, const cv::Rect& rect) const{
		cv::Mat current_shape;
		initial_shape.copyTo(current_shape);

		std::vector<float> feature_pixel_values;
		for (unsigned long iter = 0; iter < forests.size(); ++iter){
			extract_feature_pixel_values(img, rect, current_shape, initial_shape, anchor_idx[iter], deltas[iter], feature_pixel_values);
			// evaluate all the trees at this level of the cascade.
			for (unsigned long i = 0; i < forests[iter].size(); ++i)
				current_shape += forests[iter][i](feature_pixel_values);
		}
		// convert the current_shape into a full_object_detection
		obj o;
		o.r = rect;
		const Eigen::Matrix<float, 2, 3> tform_to_img = unnormalizing_tform(rect);
		o.parts.resize(current_shape.cols / 2);
		for (unsigned long i = 0; i < o.parts.size(); ++i){
			Eigen::Vector2f p = location(current_shape, i);
			Eigen::Vector3f x(p(0), p(1), 1);
			o.parts[i] = tform_to_img * x;
		}
		return o;
	}

	friend void serialize(const shape_predictor& item, std::ostream& out){
		int version = 1;
		serialize(version, out);
		serialize(item.initial_shape, out);
		serialize(item.forests, out);
		serialize(item.anchor_idx, out);
		serialize(item.deltas, out);
	}

	friend void deserialize(shape_predictor& item, std::istream& in) {
		int version = 0;
		deserialize(version, in);
		deserialize(item.initial_shape, in);
		deserialize(item.forests, in);
		deserialize(item.anchor_idx, in);
		deserialize(item.deltas, in);
	}

private:
	cv::Mat initial_shape;
	vector<vector<regression_tree> > forests;
	vector<vector<unsigned long> > anchor_idx;
	vector<vector<Eigen::Vector2f > > deltas;
};

class shape_predictor_trainer{
public:
	shape_predictor_trainer(){
		_cascade_depth = 10; _tree_depth = 4; _num_trees_per_cascade_level = 500; _nu = 0.1;
		_oversampling_amount = 20; _feature_pool_size = 400; _lambda = 0.1;
		_num_test_splits = 20; _feature_pool_region_padding = 0; _verbose = false;
	}
	unsigned long get_cascade_depth() const { return _cascade_depth; }
	void set_cascade_depth(unsigned long depth){ _cascade_depth = depth; }
	unsigned long get_tree_depth() const { return _tree_depth; }
	void set_tree_depth(unsigned long depth){ _tree_depth = depth; }
	unsigned long get_num_trees_per_cascade_level() const { return _num_trees_per_cascade_level; }
	void set_num_trees_per_cascade_level(unsigned long num){ _num_trees_per_cascade_level = num; }
	double get_nu() const { return _nu; }
	void set_nu(double nu){ _nu = nu; }
	std::string get_random_seed() const { return rnd.get_seed(); }
	void set_random_seed(const std::string& seed) { rnd.set_seed(seed); }
	unsigned long get_oversampling_amount() const { return _oversampling_amount; }
	void set_oversampling_amount(unsigned long amount){ _oversampling_amount = amount; }
	unsigned long get_feature_pool_size() const { return _feature_pool_size; }
	void set_feature_pool_size(unsigned long size){ _feature_pool_size = size; }
	double get_lambda() const { return _lambda; }
	void set_lambda(double lambda){ _lambda = lambda; }
	unsigned long get_num_test_splits() const { return _num_test_splits; }
	void set_num_test_splits(unsigned long num){ _num_test_splits = num; }
	double get_feature_pool_region_padding() const { return _feature_pool_region_padding; }
	void set_feature_pool_region_padding(double padding){ _feature_pool_region_padding = padding; }
	void be_verbose(){ _verbose = true; }
	void be_quiet(){ _verbose = false; }

	shape_predictor train(const std::vector<cv::Mat> images, const std::vector<std::vector<obj> >& objects) const{
		unsigned long num_parts = 0;
		for (unsigned long i = 0; i < objects.size(); ++i){
			for (unsigned long j = 0; j < objects[i].size(); ++j){
				if (num_parts == 0){
					num_parts = objects[i][j].parts.size();
				}
			}
		}
		rnd.set_seed(get_random_seed());
		vector<training_sample> samples;
		const cv::Mat initial_shape = populate_training_sample_shapes(objects, samples);
		const vector<vector<Eigen::Vector2f > > pixel_coordinates = randomly_sample_pixel_coordinates(initial_shape);

		unsigned long trees_fit_so_far = 0;
		if (_verbose)  std::cout << "Fitting trees..." << std::endl;

		vector<vector<regression_tree> > forests(get_cascade_depth());
		// Now start doing the actual training by filling in the forests
		for (unsigned long cascade = 0; cascade < get_cascade_depth(); ++cascade){
			// Each cascade uses a different set of pixels for its features.  We compute
			// their representations relative to the initial shape first.
			vector<unsigned long> anchor_idx;
			vector<Eigen::Vector2f > deltas;
			create_shape_relative_encoding(initial_shape, pixel_coordinates[cascade], anchor_idx, deltas);
			// First compute the feature_pixel_values for each training sample at this
			// level of the cascade.
			for (unsigned long i = 0; i < samples.size(); ++i){
				extract_feature_pixel_values(images[samples[i].image_idx], samples[i].rect,
					samples[i].current_shape, initial_shape, anchor_idx,
					deltas, samples[i].feature_pixel_values);
			}

			// Now start building the trees at this cascade level.
			for (unsigned long i = 0; i < get_num_trees_per_cascade_level(); ++i){
				forests[cascade].push_back(make_regression_tree(samples, pixel_coordinates[cascade]));
				if (_verbose){
					++trees_fit_so_far;
				}
			}
		}

		if (_verbose) std::cout << "Training complete                          " << std::endl;

		return shape_predictor(initial_shape, forests, pixel_coordinates);
	}
private:
	static cv::Mat object_to_shape(const obj& o){
		cv::Mat shape(1, o.parts.size() * 2, CV_32F);
		const  Eigen::Matrix<float, 2, 3> tform_from_img = normalizing_tform(o.r);
		for (unsigned long i = 0; i < o.parts.size(); ++i){
			Eigen::Vector2f t = o.parts[i];
			Eigen::Vector2f p = tform_from_img * Eigen::Vector3f(t(0), t(1), 1);
			shape.ptr<float>(0)[2 * i] = p(0);
			shape.ptr<float>(0)[2 * i + 1] = p(1);
		}
		return shape;
	}

	struct training_sample{
		unsigned long image_idx;
		cv::Rect rect;
		cv::Mat target_shape;
		cv::Mat current_shape;
		vector<float> feature_pixel_values;
		void swap_(training_sample& item){
			std::swap(image_idx, item.image_idx);
			std::swap(rect, item.rect);
			std::swap(target_shape, item.target_shape);
			std::swap(current_shape, item.current_shape);
			//target_shape.swap(item.target_shape);
			//current_shape.swap(item.current_shape);
			feature_pixel_values.swap(item.feature_pixel_values);
		}
	};

	regression_tree make_regression_tree(vector<training_sample>& samples, const vector<Eigen::Vector2f >& pixel_coordinates) const{
		std::deque<std::pair<unsigned long, unsigned long> > parts;
		parts.push_back(std::make_pair(0, (unsigned long)samples.size()));

		regression_tree tree;
		// walk the tree in breadth first order
		const unsigned long num_split_nodes = static_cast<unsigned long>(pow(2.0, (double)get_tree_depth()) - 1);
		vector<cv::Mat> sums(num_split_nodes * 2 + 1);

		for (unsigned long i = 0; i<num_split_nodes * 2 + 1; i++){
			sums[i].create(samples[0].target_shape.size(), CV_32F);
			sums[i].setTo(0);
		}

		for (unsigned long i = 0; i < samples.size(); ++i)
			sums[0] += samples[i].target_shape - samples[i].current_shape;

		for (unsigned long i = 0; i < num_split_nodes; ++i){
			std::pair<unsigned long, unsigned long> range = parts.front();
			parts.pop_front();

			const split_feature split = generate_split(samples, range.first, range.second, pixel_coordinates, sums[i], sums[left_child(i)],
				sums[right_child(i)]);
			tree.splits.push_back(split);
			const unsigned long mid = partition_samples(split, samples, range.first, range.second);

			parts.push_back(std::make_pair(range.first, mid));
			parts.push_back(std::make_pair(mid, range.second));
		}

		tree.leaf_values.resize(parts.size());
		for (unsigned long i = 0; i < parts.size(); ++i){
			if (parts[i].second != parts[i].first)
				tree.leaf_values[i] = sums[num_split_nodes + i] * get_nu() / (parts[i].second - parts[i].first);
			else
				tree.leaf_values[i].setTo(0);
			// now adjust the current shape based on these predictions
			for (unsigned long j = parts[i].first; j < parts[i].second; ++j)
				samples[j].current_shape += tree.leaf_values[i];
		}
		return tree;
	}

	split_feature randomly_generate_split_feature(const vector<Eigen::Vector2f >& pixel_coordinates) const{
		const double lambda = get_lambda();
		split_feature feat;
		double accept_prob;
		do{
			feat.idx1 = rnd.get_random_32bit_number() % get_feature_pool_size();
			feat.idx2 = rnd.get_random_32bit_number() % get_feature_pool_size();
			const double dist = length_(pixel_coordinates[feat.idx1] - pixel_coordinates[feat.idx2]);
			accept_prob = exp(-dist / lambda);
		} while (feat.idx1 == feat.idx2 || !(accept_prob > rnd.get_random_double()));

		feat.thresh = (rnd.get_random_double() * 256 - 128) / 2.0;
		return feat;
	}

	split_feature generate_split(const vector<training_sample>& samples, unsigned long begin,
		unsigned long end, const vector<Eigen::Vector2f >& pixel_coordinates,
		const cv::Mat& sum, cv::Mat& left_sum, cv::Mat& right_sum) const{
		// generate a bunch of random splits and test them and return the best one.
		const unsigned long num_test_splits = get_num_test_splits();
		// sample the random features we test in this function
		vector<split_feature> feats;
		feats.reserve(num_test_splits);
		for (unsigned long i = 0; i < num_test_splits; ++i)
			feats.push_back(randomly_generate_split_feature(pixel_coordinates));

		vector<cv::Mat > left_sums(num_test_splits);
		vector<unsigned long> left_cnt(num_test_splits);
		// now compute the sums of vectors that go left for each feature
		cv::Mat temp;
		for (unsigned long j = begin; j < end; ++j){
			temp = samples[j].target_shape - samples[j].current_shape;
			for (unsigned long i = 0; i < num_test_splits; ++i){
				if (samples[j].feature_pixel_values[feats[i].idx1] - samples[j].feature_pixel_values[feats[i].idx2] > feats[i].thresh){
					left_sums[i] += temp;
					++left_cnt[i];
				}
			}
		}
		// now figure out which feature is the best
		double best_score = -1;
		unsigned long best_feat = 0;
		for (unsigned long i = 0; i < num_test_splits; ++i){
			// check how well the feature splits the space.
			double score = 0;
			unsigned long right_cnt = end - begin - left_cnt[i];
			if (left_cnt[i] != 0 && right_cnt != 0){
				temp = sum - left_sums[i];

				score = left_sums[i].dot(left_sums[i]) / left_cnt[i] + temp.dot(temp) / right_cnt;
				if (score > best_score){
					best_score = score;
					best_feat = i;
				}
			}
		}
		swap(left_sums[best_feat], left_sum);
		if (left_sum.cols != 0){
			right_sum = sum - left_sum;
		}
		else{
			right_sum = sum;
			left_sum.setTo(0);
		}
		return feats[best_feat];
	}

	unsigned long partition_samples(const split_feature& split, vector<training_sample>& samples,
		unsigned long begin, unsigned long end) const{
		unsigned long i = begin;
		for (unsigned long j = begin; j < end; ++j){
			if (samples[j].feature_pixel_values[split.idx1] - samples[j].feature_pixel_values[split.idx2] > split.thresh){
				samples[i].swap_(samples[j]);
				++i;
			}
		}
		return i;
	}

	cv::Mat populate_training_sample_shapes(const vector<vector<obj> >& objects, vector<training_sample>& samples) const{
		samples.clear();
		cv::Mat mean_shape;
		long count = 0;
		// first fill out the target shapes
		for (unsigned long i = 0; i < objects.size(); ++i){
			for (unsigned long j = 0; j < objects[i].size(); ++j){
				training_sample sample;
				sample.image_idx = i;
				sample.rect = objects[i][j].r;
				sample.target_shape = object_to_shape(objects[i][j]);
				for (unsigned long itr = 0; itr < get_oversampling_amount(); ++itr)
					samples.push_back(sample);
				mean_shape += sample.target_shape;
				++count;
			}
		}
		mean_shape /= count;
		// now go pick random initial shapes
		for (unsigned long i = 0; i < samples.size(); ++i){
			if ((i%get_oversampling_amount()) == 0){
				samples[i].current_shape = mean_shape;
			}
			else{
				const unsigned long rand_idx = rnd.get_random_32bit_number() % samples.size();
				const unsigned long rand_idx2 = rnd.get_random_32bit_number() % samples.size();
				const double alpha = rnd.get_random_double();
				samples[i].current_shape = alpha*samples[rand_idx].target_shape + (1 - alpha)*samples[rand_idx2].target_shape;
			}
		}

		return mean_shape;
	}

	void randomly_sample_pixel_coordinates(std::vector<Eigen::Vector2f >& pixel_coordinates, const double min_x,
		const double min_y, const double max_x, const double max_y) const{
		pixel_coordinates.resize(get_feature_pool_size());
		for (unsigned long i = 0; i < get_feature_pool_size(); ++i){
			pixel_coordinates[i].x() = rnd.get_random_double()*(max_x - min_x) + min_x;
			pixel_coordinates[i].y() = rnd.get_random_double()*(max_y - min_y) + min_y;
		}
	}

	std::vector<std::vector<Eigen::Vector2f > > randomly_sample_pixel_coordinates(const cv::Mat& initial_shape) const{
		const double padding = get_feature_pool_region_padding();
		float min_x = FLT_MAX, min_y = FLT_MAX, max_x = -FLT_MAX, max_y = -FLT_MAX;
		const float* ptr = initial_shape.ptr<float>(0);
		for (int i = 0; i<initial_shape.cols / 2; i++){
			if (ptr[2 * i]<min_x) min_x = ptr[2 * i];
			if (ptr[2 * i]>max_x) max_x = ptr[2 * i];
			if (ptr[2 * i + 1]<min_y) min_y = ptr[2 * i + 1];
			if (ptr[2 * i + 1]>max_y) max_y = ptr[2 * i + 1];
		}
		min_x -= padding; min_y -= padding; max_x += padding; max_y += padding;

		std::vector<std::vector<Eigen::Vector2f > > pixel_coordinates;
		pixel_coordinates.resize(get_cascade_depth());
		for (unsigned long i = 0; i < get_cascade_depth(); ++i)
			randomly_sample_pixel_coordinates(pixel_coordinates[i], min_x, min_y, max_x, max_y);
		return pixel_coordinates;
	}

	mutable class rand rnd;
	unsigned long _cascade_depth, _tree_depth, _num_trees_per_cascade_level;
	unsigned long _oversampling_amount, _feature_pool_size, _num_test_splits;
	double _nu, _lambda, _feature_pool_region_padding;
	bool _verbose;
};


