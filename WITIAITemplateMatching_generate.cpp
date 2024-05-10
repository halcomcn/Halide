// windows-20240425
// WITIAITemplateMatching_generate -o . -g TemplateWitiai -f TemplateWitiai -e static_library,h,schedule target=host
// WITIAITemplateMatching_generate -o . -g TemplateWitiai -f TemplateWitiai -e static_library,h,schedule -p autoschedule_mullapudi2016.dll target=host autoscheduler=Mullapudi2016 autoscheduler.parallelism=32 autoscheduler.last_level_cache_size=16777216 autoscheduler.balance=40
#include "Halide.h"
#include <stdio.h>
using namespace Halide;
// We will define a generator to auto-schedule.
class TemplateWitiaiMain : public Halide::Generator<TemplateWitiaiMain> {
public:
	Input<Buffer<uint8_t, 3>> input1{ "input1" };
	Input<Buffer<uint8_t, 3>> templ{ "templ" };
	Output<Buffer<float, 2>> output1{ "output1" };
	Output<Buffer<float, 2>> output2{ "output2" };
	void generate()
	{
		//input1.dim(0).set_stride(3);  // stride in dimension 0 (x) is three
		//input1.dim(1).set_stride(3);  // stride in dimension 1 (y) is three
		input1.dim(0).set_bounds(0, input1.width());
		input1.dim(1).set_bounds(0, input1.height());
		//input1_1(x, y) = input1(x, y, 0);
		//input_16(x, y) = cast<uint16_t>(input1_1(x, y));
		//limit = BoundaryConditions::constant_exterior(input_16, 0, 0, input1.width(), 0, input1.height());
		limit = BoundaryConditions::repeat_edge(input1);
		input1_1(x, y) = limit(x, y, 0);
		input_16(x, y) = cast<float>(input1_1(x, y));
		templ_1(x, y) = templ(x, y, 0);
		temp_16(x, y) = cast<float>(templ_1(x, y));
		RDom matchDom(0, templ.width(), 0, templ.height());
		//matchDom.where(matchDom.x + matchDom.y > 25);  // 增大步长
		//Expr strided_x = x * 2;
		//Expr strided_y = x * 2;
		score(x, y) = sum(matchDom, pow(temp_16(matchDom.x, matchDom.y) - input_16(x * 3 + matchDom.x, y * 3 + matchDom.y), 2));
		//
		RDom searchDom(0, input1.width() / 3 - templ.width(), 0, input1.height() / 3 - templ.height());
		Tuple searchBest = argmin(searchDom, score(searchDom.x, searchDom.y), "argmin");
		//searchBest = argmin(searchDom, score(searchDom.x, searchDom.y), "argmin");
		//Func bestX;
		bestX(x, y) = searchBest[0];
		//Func bestY;
		bestY(x, y) = searchBest[1];
		//Realization re = best.realize();
		//Buffer<int> x_coordinate(re[0]);
		//Buffer<int> y_coordinate(re[1]);
		//int bestX0 = bestX(0);
		//int bestY0 = bestY(0);

		//output1(x, y) = cast<uint8_t>(best(x, y));
		output1(x, y) = cast<float>(bestX(x, y));
		output2(x, y) = cast<float>(bestY(x, y));
	}

	void schedule()
	{
		if (using_autoscheduler())
		{
			// 
		}
		else
		{
			limit.compute_root();
			input1_1.compute_root();
			input_16.compute_root();
			templ_1.compute_root();
			temp_16.compute_root();
			score.vectorize(x, 8).parallel(y).compute_root();
			//score.compute_root();
			/*bestX.compute_root();
			bestY.compute_root();*/
			/*		bestX.parallel(x).parallel(y).compute_root();
					bestY.parallel(x).parallel(y).compute_root();*/
					/*bestX.vectorize(x, 16).parallel(y).compute_root();
					bestY.vectorize(x, 16).parallel(y).compute_root();*/
			bestX.vectorize(x, 16).parallel(y).compute_root();
			bestY.vectorize(x, 16).parallel(y).compute_root();
			//bestX.tile(x, y, x_outer, y_outer, x_inner, y_inner, 64, 64)
			//	.fuse(x_outer, y_outer, tile_index)
			//	.vectorize(x_inner, 4)
			//	.parallel(tile_index)
			//	.compute_root();
			//bestY.tile(x, y, x_outer, y_outer, x_inner, y_inner, 64, 64)
			//	.fuse(x_outer, y_outer, tile_index)
			//	.vectorize(x_inner, 4)
			//	.parallel(tile_index)
			//	.compute_root();
	/*		output1.vectorize(x, 16).parallel(y).compute_root();
			output2.vectorize(x, 16).parallel(y).compute_root();*/
			output1.compute_root();
			output2.compute_root();
		}
	}

private:
	Var x{ "x" }, y{ "y" }, c{ "c" }, x_outer{ "x_outer" }, y_outer{ "y_outer" }, x_inner{ "x_inner" }, y_inner{ "y_inner" }, tile_index{ "tile_index" };
	Func limit, input_16, input1_1, templ_1, temp_16, score, bestX, bestY;
	//Tuple searchBest;
	//RDom matchDom, searchDom;
};

// file along with tools/GenGen.cpp.
HALIDE_REGISTER_GENERATOR(TemplateWitiaiMain, TemplateWitiai)
