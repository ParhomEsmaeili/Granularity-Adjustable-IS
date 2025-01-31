import copy 
import os 
import json
import torch 
from typing import Callable, Optional, Union

#NOTE: These implementations are fully intended for instances where point placements are VALID corresponding to the ground truth masks (i.e. no user mistakes).


def sparse_to_dense_reformat(point_sets, point_parametrisations, points_label, class_label_config):
        
        '''
        Input: 
        Set of guidance points in torch tensor format.
        Set of point parametrisations in dict[torch tensor] format separated by weightmap type.
        Set of corresponding points labels in torch tensor format.
        class label config: The dict containing the class-label and the class-codes.
        
        Returns: A class separated dict for the interaction point coordinates, and a nested dict (weightmap type -> class) for point parametrisations.
        '''
         
        if not type(class_label_config) == dict:
            raise TypeError('Class label-integer code configs are not in dict: [str, int] format')

        if not type(point_sets) == torch.Tensor: 
            raise TypeError('Points were presented in the incorrect format, should be a torch tensor with BNHWD shape (B=1 typically)')
        
        if not type(point_parametrisations) == dict[str, torch.Tensor]:
            raise TypeError('Point parametrisations were presented in the correct format, should be a dict (separated by click weightmap types) of torch tensors')
        
        if not type(points_label) == torch.Tensor: 
            raise TypeError('Points labels were presented in the incorrect format, should be a torch tensor with BN_point(N_spat_dims) shape (B=1 typically)')
        
        if point_sets.ndim != 3 or points_label.ndim!= 2:
            raise ValueError('The points set or the points label set were not formatted in the correct shape')
        
        # points_batch_list = []
        # param_batch_list = []

        # for batch_idx, point_set in enumerate(point_sets):
        points_dict = dict()
        output_param_dict = dict() 

        for class_label, integer_code in class_label_config.items():
            points_dict[class_label] = [point_set[class_pos_idx].clone().detach().tolist() for class_pos_idx in torch.argwhere(points_label[batch_idx] == integer_code)]    
        
        for weightmap_type, prompt_param_dict in point_parametrisations.items():
            param_dict = dict() 
            for class_label, integer_code in class_label_config.items():
                if points_dict[class_label] == []:
                    #Create a dummy parametrisation
                    param_dict[class_label] = [100000] * point_sets.ndim 
                else:
                    param_dict[class_label] = [prompt_param_dict[batch_idx][class_pos_idx].clone().detach().tolist() for class_pos_idx in torch.argwhere(points_label[batch_idx] == integer_code)]
            output_param_dict[weightmap_type] = param_dict
    
        # points_batch_list.append(points_dict)
        # param_batch_list.append(output_param_dict) 

        # return points_batch_list, param_batch_list 
        return points_dict, output_param_dict
def prompt_to_metric_granularity_converter(points, input_granularity_prompt, weightmap_types, gt):
    raise NotImplementedError('Not required.')
    #Intended for the conversion of spatial granularity maps from input to output masks.
    '''
    Inputs:
    points: A batchwise list of class-separated guidance point dicts
    input_granularity_prompt: A batchwise list of nested dicts: weightmap_type separated -> class separated guidance point parametrisations ()
    weightmap_mapping: A dict of weightmap types (separated by input and output) used for generation of the masks, which may require reparametrisation
    gt: A batchwise torch tensor for the ground truth label 

    outputs: A batchwise list of parametrisations for the metric mask (or for a loss) [A nested list of structure: weightmap_type -> class label -> point coord correspondence]
    '''
    binarised_granularity_maps = ['Ellipsoid', 'Cuboid', '2D Intersections'] 
    non_binarised_granularity_maps = ['Exponentialised Scaled Euclidean Distance']

    spatial_maps = ['Ellipsoid', 'Cuboid', 'Exponentialised Scaled Euclidean Distance']

    #For the binarised granularity maps, there will be no mapping/conversion. A spheroid of axial dims/params will be mapped to a cuboid with the same params for the dim length/2
    #Mapping is not commutative wrt binarisation. Only non-binarised weightmap types can be mapped to a binarised map for their parametrisation.

    #Checking all weightmaps belong to the entire set of valid options
    if not any([i not in non_binarised_granularity_maps + binarised_granularity_maps for i in [j for js in [list(x) for x in weightmap_types.values()] for j in js]]):
        raise ValueError('There is a selected weightmap type not supported')

    if any([i in non_binarised_granularity_maps for i in weightmap_types['output']]):
        raise ValueError('Only binarised granularity maps are permitted for output conversion')
     
    #Assert that only one of the spatial maps (ellipsoid or cuboid or euclidena distance) can be used at a given time for input or output.

    if len(set(weightmap_types['input']) and set(spatial_maps)) != 1:
        raise ValueError('There should be exactly one spatial map for input conversion')
    
    if len(set(weightmap_types['output']) and set(spatial_maps)) != 1:
        raise ValueError('There should only be one spatial map for the output.')

    #Performing the conversions.

    if any([i in non_binarised_granularity_maps for i in weightmap_types['input']]):
        raise NotImplementedError('Conversion from non-binarised input weightmap type to binarised metric/loss mask is not yet supported, requires functions which map this conversion')
    
    if 'Ellipsoid' in weightmap_types['input'] and 'Ellipsoid' in weightmap_types['output']:
        pass 
    elif 'Cuboid' in weightmap_types['input'] and 'Cuboid' in weightmap_types['output']:
        pass 

    elif 'Cuboid' in weightmap_types['input'] and 'Ellipsoid' in weightmap_types['output']:
        metric_granularity_parametrisation = copy.deepcopy(input_granularity_prompt) 
        metric_granularity_parametrisation['Ellipsoid'] = input_granularity_prompt['Cuboid']
        del metric_granularity_parametrisation['Cuboid']

    elif 'Ellipsoid' in weightmap_types['input'] and 'Cuboid' in weightmap_types['output']:
        metric_granularity_parametrisation = copy.deepcopy(input_granularity_prompt)
        metric_granularity_parametrisation['Cuboid'] = input_granularity_prompt['Ellipsoid']
        del metric_granularity_parametrisation['Ellipsoid']

    return metric_granularity_parametrisation

def sparse_granularity_heuristic(weightmap_types,
                                prev_mask,
                                gt
                                ):
    NotImplementedError('Need to implement the heuristic which samples the spatial granularity wrt point placement and error region.')

def sparse_granularity_parametriser(weightmap_types: list[str], 
                                    param_heuristic_bool: bool,
                                    points_set: list[torch.Tensor],
                                    points_label_set: list[torch.Tensor], 
                                    scribbles_set: list[torch.Tensor], 
                                    scribbles_label_set: list[torch.Tensor],
                                    prev_mask: Union[torch.Tensor, None],
                                    gt: Union[torch.Tensor, None],
                                    guidance_parametrisation: Union[dict[str, dict[str,list]], None]):

    '''
    Input:

    Points_set: List of point coord tensors (1 x 1 x N_dim)
    Points Label set: List of point label tensors (1 x 1)
    Scribbles_set: List of scribble tensors for each scribble (1 x N_scrib_p x N_dim)
    Scribbles Label Set: List of scribble label tensors for each scribble of shape (1 x N_scrib_p)
    weightmap_types: The types of the spatial weightmaps (implicitly for sparse parametrisation) being used
    param_heuristic_bool: Whether the parametrisation needs to be determined heuristically, or is already provided.
    prev_mask: torch tensor 1 x HWD
    gt: optional gt for heuristic torch tensor 1 x HWD
    guidance parametrisation: optional nested dict separated by point/scribble, weightmap type, then a fixed parametrisation for that given config... 

    Return:

    merged_points: The merged points across the point and scribble sets into one tensor (1 x N_p x N_dim)
    merged_labels: The merged labels across the point and scribble sets into one tensor (1 x N_p) 
    output_guidance_parametrisations: A dictionary, separated by weightmap type, of the point parametrisations in tensor form (1 x N_p x N_params)
    '''

    if param_heuristic_bool:
        #In this case, a heuristic is being used to generate the granularity prompt.
        # Heuristic should generate a parametrisation tensor which matches the exact structure of the corresponding prompt (but not the spatial dims)
        # I.e. for points_set: list[] 
        raise NotImplementedError('Need to implement the heuristic for generating the spatial granularity size')
    else:
        #If no heuristic is used, then fixed granularity is being used and applied uniformly.
        if guidance_parametrisation == None:
            raise ValueError('The guidance parametrisation must exist in the absence of a heuristic generation method.')
      
    
    #We reformat the prompt coordinate information. 

    if not scribbles_set:
        #Empty list evaluates as a false
        no_scribble=True
        if scribbles_label_set:
            raise ValueError('There should not be a scribbles label set if there is not a scribbles set')
    else:
        
        if not scribbles_label_set:
            raise ValueError('There should be a scribbles label set if there is a scribbles set')


        if len(scribbles_set) > 1:
            #If quantity of scribbles > 1
            scribble_set_reformat = torch.cat(scribbles_set, dim=1)
            scribble_label_reformat = torch.cat(scribbles_label_set, dim=1)
            no_scribble = False 

        elif len(scribbles_set) == 1:
            #If batch size = 1 then we use the single scribble tensor instead. 
            scribble_set_reformat = scribbles_set[0]
            scribble_label_reformat = scribbles_label_set[0]
            no_scribble = False
        

    #We do the same thing for the points, these are a list of 1 x Ndim tensors.

    if not points_set:
        no_points = True
        if points_label_set:
            raise ValueError('There should not be a points label set if there is no points set')
    else:
        if not points_label_set:
            raise ValueError('There should be a points label set if there is a points set')

        if len(points_set) > 1:
            #Num points > 1:
            point_set_reformat = torch.cat(points_set, dim=1)
            point_label_reformat = torch.cat(points_label_set, dim=1)
            no_points=False 
            
        elif len(points_set) == 1:
            point_set_reformat = points_set[0]
            point_label_reformat = points_label_set[0]
            no_points = False
    

    if no_points and no_scribble:
        raise ValueError('No points or scribbles provided.')
    

    #Create a guidance parametrisation dict. 
    output_guidance_parametrisations = dict() 

    for weightmap in weightmap_types:

    
        if param_heuristic_bool:
            #We will implement this through tensor multiplication, the assumption is that the parametrisations are already reformatted.
            raise NotImplementedError('Need to implement the reformatting of the parametrisations here. Should be in dict[weightmap_type[list reformat')
            assert type(heuristic_guidance_parametrisation) == dict 

        else: 
            #We apply a fixed parametrisation uniformly across the set of points and scribbles, according the parametrisation for points, and the parametrisation for scribbles.
            #NOTE: We parametrise the points and scribbles separately, in case they have separate parametrisations
            if not no_points:
                point_parametrisations = torch.ones((point_set_reformat.shape[0], point_set_reformat.shape[1], len(guidance_parametrisation['points'][weightmap]))).to(device=point_set_reformat.device)
                point_parametrisations *= torch.tensor(guidance_parametrisation['points'][weightmap]).to(device=point_set_reformat.device) #Broadcast the parametrisation for the points 

            if not no_scribble:
                scribble_parametrisations = torch.ones((scribble_set_reformat.shape[0], scribble_set_reformat.shape[1], len(guidance_parametrisation['scribbles'][weightmap]))).to(device=scribble_set_reformat.device)
                scribble_parametrisations *= torch.tensor(guidance_parametrisation['scribbles'][weightmap]).to(device=scribble_set_reformat.device) 
                #We broadcast the parametrisation according to the number of parameters in the parametrisation to the scribble points.


            if not no_points and not no_scribble:
                #We merge the sparse parametrisations together.
                merged_point_parametrisations = torch.cat([point_parametrisations, scribble_parametrisations], dim=1)

            elif not no_points and no_scribble:
                merged_point_parametrisations = point_parametrisations 

            elif no_points and not no_scribble:
                merged_point_parametrisations = scribble_parametrisations 

            output_guidance_parametrisations[weightmap] = merged_point_parametrisations 

    # Merging the points:

    if not no_points and not no_scribble:
        #We merge the set of points together, and also merge the sparse parametrisations together.
        merged_points = torch.cat([point_set_reformat, scribble_set_reformat], dim=1)
        merged_point_labels = torch.cat([point_label_reformat, scribble_label_reformat], dim=1)

    elif not no_points and no_scribble:

        merged_points = point_set_reformat
        merged_point_labels = point_label_reformat

    elif no_points and not no_scribble:
        merged_points = scribble_set_reformat
        merged_point_labels = scribble_label_reformat
        
    return merged_points, merged_point_labels, output_guidance_parametrisations 

def reformat_prompt_and_gen_param(
                        sparse_dense: str,
                        param_heuristic_bool: bool,
                        weightmap_types: list[str],
                        points_set: Optional[list[torch.Tensor]] = None,
                        scribbles_set: Optional[list[torch.Tensor]] = None,
                        points_label_set: Optional[list[torch.Tensor]] = None,
                        scribbles_label_set: Optional[list[torch.Tensor]] = None, 
                        guidance_parametrisation: Optional[dict[str,dict[str, torch.Tensor]]] = None,
                        prev_mask: Optional[torch.Tensor] = None,
                        gt: Optional[torch.Tensor] = None,
                        class_config: Optional[dict[str,int]] = None):
    '''

    This function is intended for the generation of prompt parametrisations for inference, points and scribbles are optionally separated in the instance where one would
    like to have the option treat these prompts distinctly. 

    Inputs:
    
    sparse_dense: str which indicates whether the prompts granularity encoding is sparse or dense (OR BOTH required) (sparse = param input, dense = voxel-wise map)

    param_heuristic_bool: Bool which indicates whether a heuristic is to be used for generating the granularity param.

    weightmap_types: Required for the heuristic granularity prompt generator such that the dict is appropriately structured.. 

    points: Optional list of points for a given image sample in a batch, each point is of form: torch tensor (1 x 1 x N_dim)

    points_label_set: Optional list of Torch tensor with size (1 x 1) denoting the class of the points

    scribbles:  Optional (if scribble set is parametrised separately to points) list of torch tensors [N_scrib [1 x N_p x N_dim]] (torch tensor per scribble set)
    scribbles_label_set: (if scribble set is parametrised separately to poitns) list of torch tensors [N_scrib (1 x N_scrib_p)] denoting the class of the scribble points.

    guidance_parametrisations: Optional: If providing fixed parametrisations for the prompt granularity, it is a nested dict separated by weightmap type, point/scribble, and a 
    parametrisation of form torch.Tensor with shape 1 x 1 x N_params.

    prev_mask : torch tensor containing the discretised mask (shape 1 x HWD)
    gt: torch tensor containig the discrete ground truth mask (shape 1 x HWD)
    
    Returns:

    The points (merged into one torch tensor from the points and the scribbles). 
    
    Input guidance parametrisations in dict[weightmap_type[torch tensor]] format with tensors of shape (1 x (N_points + N_scrib_points) x N_params) for sparse encoding, OR 
    dict[weightmap_type[class, list[individual point_parametrisations]] format for dense maps encoding] (WITH DUMMY POINTS FOR EMPTY CLASSES!).


    Potential uses cases: 
    1) generation of sparse granularity params (only intended for input), 
    2) generation of dense granularity params for input OR output
    3) the generation of sparse AND dense granularity params, in this case the assumption is that they must match wrt parametrisation (either for the fixed params a prior, 
    or we carry the heuristic parametrisations over for each point/scribble)

    Note: This allows for the following generations: 
    1) Disentangled generation of input and output granularity params with fixed param (by calling the func separately)
    2) Disentangled generation of input OR output granularity params with a fixed param (by calling the func separately)
    3) The generation of sparse and also optionally dense granularity params with a consistent parametrisation generated by a heuristic (consistent between input - output)
    
    This allows the following cross exams: 
    1) Input and output granularity match (fixed param) (generated with a single, or optionally separate calls of the func)
    2) Input and output granularity don't match (fixed param) (generated with separate calls of the func)
    3) Heuristically generated sparse and optionally dense granularity params which can match for input and output (must match with weightypes!) (single call of the func)
    4) Heuristically generated input params, and fixed output granularity which doesn't match (generated with separate calls of the func).
    '''

    #Create a list of binary weightmaps for granularity prompting strategies (and for denoting the intent of sparse ones too):
    binary_weightmaps = ['Ellipsoid', 'Cuboid']

    if any([weightmap not in binary_weightmaps for weightmap in weightmap_types]):
        raise KeyError('The weightmap selected is not supported')
    
    #Generation of the parametrisations: #All parametrisations will be saved under a dict, separated by the weightmap type as the key. The only distinction between 
    #sparse and dense is the structure within the weightmap type (i.e. torch tensor vs nested lists).

    merged_points, merged_point_labels, merged_point_parametrisations = sparse_granularity_parametriser(weightmap_types,
                                                                                                        param_heuristic_bool, 
                                                                                                        points_set,
                                                                                                        points_label_set,
                                                                                                        scribbles_set,
                                                                                                        scribbles_label_set,
                                                                                                        prev_mask,
                                                                                                        gt,
                                                                                                        guidance_parametrisation
    )

    if sparse_dense.title() == 'Sparse':
        #Only generating parametrisation (only potentially applicable for the input), we do nothing.
        return {'sparse_points':merged_points, 'sparse_parametrisations':merged_point_parametrisations, 'sparse_labels':merged_point_labels}
    elif sparse_dense.title() == 'Dense':
        #Only generating dense parametrisations: 

        #Here we will generate the dict format expected for the dense mask generator. Structure is as follows: dict[weightmap_type[class[points/params]]] 

        #This is done by mapping the sparse parametriser to a dense format using the reformatter structure. 
        points_dict, parametrisation_dict = sparse_to_dense_reformat(merged_points, merged_point_parametrisations, merged_point_labels, class_config) 
        return {'dense_points':points_dict, 'dense_parametrisations':parametrisation_dict}
    
    elif sparse_dense.title() == 'Both':
        #NOTE: In instances where we generate an input granularity AND a mask parametrisation which is used for both (in common), we use the reformatter and output both.
        points_dict, parametrisation_dict = sparse_to_dense_reformat(merged_points, merged_point_parametrisations, merged_point_labels, class_config) 
        return {'sparse_points': merged_points, 
                'sparse_parametrisations':merged_point_parametrisations, 
                'sparse_labels':merged_point_labels,
                'dense_points':points_dict, 
                'dense_parametrisations':parametrisation_dict}
    else: 
        raise ValueError('It has to be dense and/or sparse prompt encoding of the granularity')

    # return merged_points, merged_point_labels, merged_point_parametrisations 


if __name__ == '__main__':

    batchwise_points_list = [torch.ones([10,3])]
    batchwise_scribbles_list = [[torch.ones(10,3), torch.ones(10,3), torch.ones(10,3)]] 
    # batchwise_points