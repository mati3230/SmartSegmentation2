#include <boost/python/numpy.hpp>
#include <iostream>
//#include <tuple>

#define PCL_NO_PRECOMPILE
// basic pcl point cloud types
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>

// ransac
#include <pcl/sample_consensus/sac_model_plane.h>

// euclidean cluster extraction
#include <pcl/segmentation/extract_clusters.h>

// vccs
#include <pcl/segmentation/supervoxel_clustering.h>

#include <pcl/kdtree/kdtree.h>
// #define DEBUG

namespace p = boost::python;
namespace np = boost::python::numpy;
using namespace std;
using namespace pcl;
using namespace Eigen;

#define COLOR
#ifdef COLOR
//typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointXYZRGB PointT;
#else
typedef pcl::PointXYZ PointT;
#endif
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointXYZL PointLT;
typedef pcl::PointCloud<PointLT> PointLCloudT;

void _grow(PointCloudT::Ptr cloud, vector<int>& segments_r, int query_idx, search::KdTree<PointT>::Ptr tree, float radius, int segment){
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    #ifdef DEBUG
    cout << "get query point" << endl;
    #endif
    PointT searchPoint = cloud->points[query_idx];
    segments_r[query_idx] = segment;

    #ifdef DEBUG
    cout << "apply radius search" << endl;
    #endif
    if ( tree->radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
    {
        for (std::size_t i = 0; i < pointIdxRadiusSearch.size (); ++i){
            int cloud_idx = pointIdxRadiusSearch[i];
            if (cloud_idx == query_idx) continue;
            if (segments_r[cloud_idx] != 0) continue;
            //PointT p = cloud->points[ pointIdxRadiusSearch[i] ];
            //std::cout << "    "  <<   p.x << " " << p.y << " " << p.z << " (squared distance: " << pointRadiusSquaredDistance[i] << ")" << std::endl;
            _grow(cloud, segments_r, cloud_idx, tree, radius, segment);
        }
    }
}

void _region_growing_radius(PointCloudT::Ptr cloud, vector<int>& segments_r, float radius=0.5){
    #ifdef DEBUG
    cout << "resize list" << endl;
    #endif
    int n_points = cloud->points.size();
    segments_r.resize(n_points);
    for (int i = 0; i < n_points; i++){
        segments_r[i] = 0;
    }

    #ifdef DEBUG
    cout << "create kdtree" << endl;
    #endif
    search::KdTree<PointT>::Ptr tree (new search::KdTree<PointT>);
    #ifdef DEBUG
    cout << "set input cloud of kdtree" << endl;
    #endif
    // TODO: fix crash
    tree->setInputCloud (cloud);
    int segment = 0;

    #ifdef DEBUG
    cout << "start to grow region" << endl;
    #endif
    for (int i = 0; i < n_points; i++){
        if (segments_r[i] != 0) continue;
        segment++;
        _grow(cloud, segments_r, i, tree, radius, segment);
    }
}

bool _vccs(PointCloudT::Ptr cloud, map<int, vector<int>> &segmentsToPointIdxs, map<int, vector<int>> &segmentNeighbours, float voxel_resolution = 0.1f, float seed_resolution = 1.0f, float color_importance = 0.2f, float spatial_importance = 0.4f, float normal_importance = 1.0f){
    SupervoxelClustering<PointT> super (voxel_resolution, seed_resolution);
    super.setInputCloud (cloud);
    super.setColorImportance (color_importance);
    super.setSpatialImportance (spatial_importance);
    super.setNormalImportance (normal_importance);
    map <uint32_t, Supervoxel<PointT>::Ptr > supervoxel_clusters;
    #ifdef DEBUG
    cout << "start cluster extraction" << endl;
    #endif
    super.extract (supervoxel_clusters);
    #ifdef DEBUG
    cout << "cluster extraction done" << endl;
    #endif

    PointLCloudT::Ptr labeled_cloud = super.getLabeledCloud ();

    #ifdef DEBUG
    cout << labeled_cloud->points.size() << endl;
    #endif

    for (int i = 0; i < labeled_cloud->points.size(); i++){
        int s = labeled_cloud->points[i].label;
        // if segment s not in segmentsToPointIdxs
        if(segmentsToPointIdxs.find(s) == segmentsToPointIdxs.end()){
            vector<int> idxs;
            segmentsToPointIdxs.insert(pair<int, vector<int>>(s, idxs));
        }
        // add point idx i to list of segment s
        segmentsToPointIdxs[s].push_back(i);
    }

    std::multimap<std::uint32_t, std::uint32_t> supervoxel_adjacency;
    super.getSupervoxelAdjacency (supervoxel_adjacency);

    for (auto label_itr = supervoxel_adjacency.cbegin (); label_itr != supervoxel_adjacency.cend (); )
    {
        //First get the label
        std::uint32_t supervoxel_label = label_itr->first;
        vector<int> neighbours;
        // for all neighbours of segment supervoxel_label
        for (auto adjacent_itr = supervoxel_adjacency.equal_range (supervoxel_label).first; adjacent_itr!=supervoxel_adjacency.equal_range (supervoxel_label).second; ++adjacent_itr)
        {
            // get label of segment
            std::uint32_t supervoxel_label_neighbour = adjacent_itr->second;
            //std::cout << supervoxel_label << ": " << supervoxel_label_neighbour << std::endl;
            neighbours.push_back(supervoxel_label_neighbour);

        }
        segmentNeighbours.insert(pair<int, vector<int>>(supervoxel_label, neighbours));
        label_itr = supervoxel_adjacency.upper_bound (supervoxel_label);
    }
    #ifdef DEBUG
    cout << "size of segmentNeighbours: " << segmentNeighbours.size() << endl;
    #endif

    return true;
}

bool _ransac_plane (PointCloudT::Ptr cloud, std::vector<int> &inliers_, VectorXf& model_coefficients_, int max_iterations_=1000, double threshold_=0.01, double probability_=0.9)
{
    vector<int> indices(cloud->points.size()) ;
    // indices will be row numbers of cloud
    iota (begin(indices), end(indices), 0);
    // SampleConsensusModelPlane<PointT>::Ptr sac_model_ (new SampleConsensusModelPlane<PointT>(cloud, indices, true));
    SampleConsensusModelPlane<PointT>::Ptr sac_model_ (new SampleConsensusModelPlane<PointT>(cloud, indices, false));

    int iterations_ = 0;
    int n_best_inliers_count = -INT_MAX;
    double k = 1.0;

    vector<int> selection, model_;
    VectorXf model_coefficients;

    double log_probability  = log (1.0 - probability_);
    double one_over_indices = 1.0 / static_cast<double> (sac_model_->getIndices ()->size ());

    int n_inliers_count = 0;
    unsigned skipped_count = 0;
    // suppress infinite loops by just allowing 10 x maximum allowed iterations for invalid model parameters!
    const unsigned max_skip = max_iterations_ * 10;

    // Iterate
    while (iterations_ < k && skipped_count < max_skip)
    {
        // Get X samples which satisfy the model criteria
        sac_model_->getSamples (iterations_, selection);

        if (selection.empty ())
        {
            //PCL_ERROR ("[pcl::RandomSampleConsensus::computeModel] No samples could be selected!\n");
            break;
        }

        // Search for inliers in the point cloud for the current plane model M
        if (!sac_model_->computeModelCoefficients (selection, model_coefficients))
        {
            //++iterations_;
            ++skipped_count;
            continue;
        }

        // Select the inliers that are within threshold_ from the model
        //sac_model_->selectWithinDistance (model_coefficients, threshold_, inliers);
        //if (inliers.empty () && k > 1.0)
        //  continue;

        n_inliers_count = sac_model_->countWithinDistance (model_coefficients, threshold_);

        // Better match ?
        if (n_inliers_count > n_best_inliers_count)
        {
            n_best_inliers_count = n_inliers_count;

            // Save the current model/inlier/coefficients selection as being the best so far
            model_coefficients_ = model_coefficients;
            model_ = selection;

            // Compute the k parameter (k=log(z)/log(1-w^n))
            double w = static_cast<double> (n_best_inliers_count) * one_over_indices;
            double p_no_outliers = 1.0 - pow (w, static_cast<double> (selection.size ()));
            p_no_outliers = (max) (numeric_limits<double>::epsilon (), p_no_outliers);       // Avoid division by -Inf
            p_no_outliers = (min) (1.0 - numeric_limits<double>::epsilon (), p_no_outliers);   // Avoid division by 0.
            k = log_probability / log (p_no_outliers);
        }

        ++iterations_;
        //PCL_DEBUG ("[pcl::RandomSampleConsensus::computeModel] Trial %d out of %f: %d inliers (best is: %d so far).\n", iterations_, k, n_inliers_count, n_best_inliers_count);
        if (iterations_ > max_iterations_)
        {
          //PCL_DEBUG ("[pcl::RandomSampleConsensus::computeModel] RANSAC reached the maximum number of trials.\n");
          break;
        }
    }

    //PCL_DEBUG ("[pcl::RandomSampleConsensus::computeModel] Model: %lu size, %d inliers.\n", model_.size (), n_best_inliers_count);

    if (model_.empty ())
    {
        inliers_.clear ();
        return (false);
    }

    // Get the set of inliers that correspond to the best model found so far
    sac_model_->selectWithinDistance (model_coefficients_, threshold_, inliers_);
    return (true);
}

bool _euclidean_cluster(
        PointCloudT::Ptr cloud,
        p::list &segment_indxs,
        float tolerance = 0.03,
        int min_cluster_size = 1000,
        int max_cluster_size = 250000
        ){
    search::KdTree<PointT>::Ptr tree (new search::KdTree<PointT>);
    tree->setInputCloud (cloud);

    EuclideanClusterExtraction<PointT> ec;
    vector<PointIndices> cluster_indices;
    ec.setClusterTolerance (tolerance);
    ec.setMinClusterSize (min_cluster_size);
    ec.setMaxClusterSize (max_cluster_size);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud);
    ec.extract (cluster_indices);

    #ifdef DEBUG
        cout << "n cluster: " << cluster_indices.size() << endl;
    #endif
    if (cluster_indices.size() > 1000 || cluster_indices.size() == 0)
        return false;

    for (vector<PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
        p::list l;
        for (vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
            l.append(*pit);
        segment_indxs.append(l);
    }
    return true;
}

void ndarrayToPCL(np::ndarray P, PointCloudT::Ptr &cloud){
    int n_points = P.shape(0);
    cloud->points.reserve(n_points);
    cloud->width=n_points;
    cloud->height=1;
    for(int i = 0; i < n_points; i++){
        PointT p;
        float x = p::extract<float>(P[i][0]);
        float y = p::extract<float>(P[i][1]);
        float z = p::extract<float>(P[i][2]);

        unsigned int r = static_cast<unsigned int>(p::extract<float>(P[i][3]));
        unsigned int g = static_cast<unsigned int>(p::extract<float>(P[i][4]));
        unsigned int b = static_cast<unsigned int>(p::extract<float>(P[i][5]));

        p.x=x;
        p.y=y;
        p.z=z;
        p.r=r;
        p.g=g;
        p.b=b;

        //cout << unsigned(p.r) << " " << unsigned(p.g) << " " << unsigned(p.b) << endl;

        cloud->points.push_back(p);
    }
    cloud->is_dense = true;
}

np::ndarray indicesToNDArray(vector<int> indices){
    Py_intptr_t shape[1] = { indices.size() };
    np::ndarray result = np::zeros(1, shape, np::dtype::get_builtin<int>());
    copy(indices.begin(), indices.end(), reinterpret_cast<int*>(result.get_data()));
    return result;
}

/*
***************************
* python interface
***************************
*/

class PCLCloud{
    private:
        PointCloudT::Ptr pcl_cloud;
        std::shared_ptr<np::ndarray> np_cloud;
        search::KdTree<PointT>::Ptr tree;
    public:
        PCLCloud(np::ndarray P);
        ~PCLCloud();
        inline PointCloudT::Ptr GetPCLCloud(){return this->pcl_cloud;}
        inline void SetNPCloud(np::ndarray P);
        inline np::ndarray GetNPCloud(){return *(this->np_cloud.get());}
        inline void BuildTree();
        inline np::ndarray Search(np::ndarray P);
};

PCLCloud::PCLCloud(np::ndarray P){
#ifdef DEBUG
    cout << "create PCLCloud" << endl;
#endif
    this->SetNPCloud(P);
#ifdef DEBUG
    cout << "done" << endl;
#endif
}

PCLCloud::~PCLCloud(){
#ifdef DEBUG
    cout << "destructor PCLCloud" << endl;
    cout << "done" << endl;
#endif
}

void PCLCloud::SetNPCloud(np::ndarray P){
    this->np_cloud = std::make_shared<np::ndarray>(P);
    this->pcl_cloud = PointCloudT::Ptr(new PointCloudT);
    ndarrayToPCL(P, this->pcl_cloud);
}

void PCLCloud::BuildTree(){
    this->tree = search::KdTree<PointT>::Ptr(new search::KdTree<PointT>);
    this->tree->setInputCloud (this->pcl_cloud);
}

np::ndarray PCLCloud::Search(np::ndarray P){
    /*
    * Insert some points P and the corresponding point indices of the points
    * will be returned if they exist in the cloud. 
    */
    if(!this->tree){
        this->BuildTree();
    }
    int n_points = P.shape(0);
    int K = 1;
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);
    std::vector<int> idxs(n_points);

    for(int i = 0; i < n_points; i++){
        float x = p::extract<float>(P[i][0]);
        float y = p::extract<float>(P[i][1]);
        float z = p::extract<float>(P[i][2]);

        unsigned int r = static_cast<unsigned int>(p::extract<float>(P[i][3]));
        unsigned int g = static_cast<unsigned int>(p::extract<float>(P[i][4]));
        unsigned int b = static_cast<unsigned int>(p::extract<float>(P[i][5]));

        PointT searchPoint;

        searchPoint.x=x;
        searchPoint.y=y;
        searchPoint.z=z;
        searchPoint.r=r;
        searchPoint.g=g;
        searchPoint.b=b;
        if (this->tree->nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
        {
            if(pointIdxNKNSearch.size() > 0){
                idxs[i] = pointIdxNKNSearch[0];
            }
        }
    }
    np::ndarray np_idxs = indicesToNDArray(idxs);
    return np_idxs;
}

p::list get_plane_segment(PCLCloud &pcl_cloud, int max_iterations, double threshold, double probability){

#ifdef DEBUG
    cout << "get plane segment (" << max_iterations << ", " << threshold << ", " << probability << ")" << endl;
#endif
    vector<int> inliers;
#ifdef DEBUG
    cout << "compute model" << endl;
#endif
    VectorXf model_coefficients_;
    bool seg_result = _ransac_plane (pcl_cloud.GetPCLCloud(), inliers, model_coefficients_, max_iterations, threshold, probability);
#ifdef DEBUG
    cout << model_coefficients_ << endl;
    cout << "done" << endl;
#endif
    np::ndarray plane = indicesToNDArray(inliers);

    p::list result;

    result.append(plane);
    result.append(model_coefficients_(0));
    result.append(model_coefficients_(1));
    result.append(model_coefficients_(2));
    result.append(model_coefficients_(3));
    result.append(seg_result);

    return result;
}

p::list euclidean_cluster(PCLCloud &pcl_cloud, float tolerance, int min_cluster_size, int max_cluster_size){
    p::list result;

    bool seg_result = _euclidean_cluster(pcl_cloud.GetPCLCloud(), result, tolerance, min_cluster_size, max_cluster_size);
    result.append(seg_result);

    return result;
}

p::list vccs(PCLCloud &pcl_cloud, float voxel_resolution, float seed_resolution, float color_importance, float spatial_importance, float normal_importance){
//p::list vccs(PCLCloud &pcl_cloud){
    #ifdef DEBUG
    cout << "start vccs" << endl;
    #endif

    map<int, vector<int>> segmentsToPointIdxs;
    map<int, vector<int>> segmentNeighbours;

    p::list seg;
    /*PointCloudT::Ptr cloud (new PointCloudT);
    if (io::loadPCDFile<PointT> ("./ScannetScenes/scene0000_00.pcd", *cloud))
    {
        return seg;
    }*/

    _vccs(pcl_cloud.GetPCLCloud(), segmentsToPointIdxs, segmentNeighbours, voxel_resolution, seed_resolution, color_importance, spatial_importance, normal_importance);
    //_vccs(pcl_cloud.GetPCLCloud(), segmentsToPointIdxs, segmentNeighbours);
    #ifdef DEBUG
    cout << "num segments: " << segmentsToPointIdxs.size() << endl;
    #endif
    for (auto it_n = segmentNeighbours.begin(); it_n != segmentNeighbours.end(); ++it_n){
        int segment = it_n->first;
        vector<int> pointIdxs = segmentsToPointIdxs[segment];
        vector<int> neighbours = it_n->second;
        np::ndarray npPointIdxs = indicesToNDArray(pointIdxs);
        np::ndarray npNeighbours = indicesToNDArray(neighbours);

        seg.append(segment);
        seg.append(npPointIdxs);
        seg.append(npNeighbours);
    }

    #ifdef DEBUG
    cout << "done" << endl;
    #endif
    return seg;
}

np::ndarray region_growing_radius(PCLCloud &pcl_cloud, float radius){
    #ifdef DEBUG
    cout << "start region growing radius" << endl;
    #endif

    vector<int> segments;
    _region_growing_radius(pcl_cloud.GetPCLCloud(), segments, radius);

    np::ndarray result = indicesToNDArray(segments);

    #ifdef DEBUG
    cout << "done" << endl;
    #endif

    return result;
}

BOOST_PYTHON_MODULE(segmentation_ext) {
    np::initialize();

    p::class_<PCLCloud>("PCLCloud", p::init<np::ndarray>())
        .def("GetNPCloud", &PCLCloud::GetNPCloud)
        .def("SetNPCloud", &PCLCloud::SetNPCloud)
        .def("BuildTree", &PCLCloud::BuildTree)
        .def("Search", &PCLCloud::Search)
    ;

    p::def("get_plane_segment", get_plane_segment);
    p::def("euclidean_cluster", euclidean_cluster);
    p::def("vccs", vccs);
    p::def("region_growing_radius", region_growing_radius);
}
