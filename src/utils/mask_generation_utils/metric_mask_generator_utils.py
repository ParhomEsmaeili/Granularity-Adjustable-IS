import torch 
import numpy as np
from itertools import chain
import operator
import functools
from monai.utils import min_version, optional_import

connected_comp_measure, _ = optional_import("skimage.measure", "0.14.2", min_version)

class MaskGenerator:
    '''
    This mask generator assumes that for mask generation with click information, that at least one click must be placed among the classes. And ideally for each of the classes (though not absolutely needed for this).
    
    Otherwise, the cross-class mask generation will completely fail.
    '''

    def __init__(self, click_map_types, gt_map_types, human_measure, dict_of_class_codes, ignore_empty, device):
        
        assert type(click_map_types) == list, "Clicking weightmap types selected were not formatted as a list"
        assert type(gt_map_types) == list, "GT-based weightmap types selected were not formatted as a list"
        assert type(human_measure) == str, "Human-centric measure selected was not formatted as a string"
        assert type(dict_of_class_codes) == dict, "Dictionary of class integer codes was not formatted as such."
        assert type(ignore_empty) == bool, "Ignore empty parameter was not provided for handling instances where click sets may not be provided for mask generation which requires it! (i.e. non-autoseg local responsiveness mask which has an empty click set)"

        self.device = device 

        self.class_code_dict = dict_of_class_codes

        self.click_weightmap_types = [i.title() for i in click_map_types]
        #A list of the components of the weight-map which may originate solely from the clicks, e.g. a distance weighting, or an ellipsoid.

        self.gt_weightmap_types = [i.title() for i in gt_map_types] 
        #A list of the components of the weight-map which may originate from the ground truth in relation to the clicks, e.g. the connected component
        
        self.human_measure = [human_measure] 
        #The measure of model performance being implemented, e.g. responsiveness in region of locality/non-worsening elsewhere. 

        self.ignore_empty = ignore_empty
        #The bool which ensures that the code can handle instances where click sets may be empty when they should not be! I.e. when generating masks for local responsiveness weightmaps where a click set may be empty.

        self.supported_click_weightmaps = ['Ellipsoid',
                                            'Cuboid', 
                                            'Scaled Euclidean Distance',
                                            'Exponentialised Scaled Euclidean Distance',
                                            'Binarised Exponentialised Scaled Euclidean Distance',
                                            '2D Intersections', 
                                            'None']
        
        self.supported_gt_weightmaps = ['Connected Component',
                                        'None']
        
        self.supported_human_measures = ['Local Responsiveness',
                                        'Temporal Non Worsening',
                                        'Temporal Consistency', #Difference is that temporal consistency has no extra input information, to filter changed voxels.
                                        'None']

        if any([click_weightmap not in self.supported_click_weightmaps for click_weightmap in self.click_weightmap_types]):
            #Basic assumption is numbers and symbols will not be placed in the string, only potentially a string with non-capitalised words.
            raise Exception("Selected click-based weight map is not supported")
        
        elif any([gt_weightmap not in self.supported_gt_weightmaps for gt_weightmap in self.gt_weightmap_types]):
            raise Exception("Selected gt-based weight map is not supported")
        
        elif any([human_measure not in self.supported_human_measures for human_measure in self.human_measure]):
            raise Exception("Selected human-centric measure is not supported")
        



    def click_based_weightmaps(self, guidance_points_set, guidance_point_parametrisations, include_background, image_dims):
        #The guidance points set is assumed to be a dictionary covering all classes, with each point being provided as a 2D/3D set of coordinates in a list.
        #Image dims are assumed to be a list.

        #Guidance point parametrisations is the nested dictionary which contains the parametrisations sorted by mask-type and then by class. 
        
        assert type(guidance_points_set) == dict, "The generation of click based weightmaps failed due to the guidance points not being in a dict"
        assert type(guidance_point_parametrisations) == dict, "The generation of click based weightmaps failed due to the guidance point parametrisations not being a dict"
        assert type(image_dims) == torch.Size, "The generation of click based weightmaps failed due to the image dimensions not being of a torch.Size class"
        assert type(include_background) == bool, "The generation of click based weightmaps failed due to the include_background parameter not being a bool: True/False"

        for value in guidance_point_parametrisations.values():
            assert type(value) == dict, "The generation of click based weightmaps failed due to the parametrisations for each weightmap_type field not being a dict"
        
        

        # list_of_points = list(chain.from_iterable(list(guidance_points_set.values())))


        click_availability_bool = list(chain.from_iterable(list(guidance_points_set.values()))) != [] #Checking that the overall click set isn't empty
        per_class_click_availability_bool = dict()    #Checking whether each click class is empty or not.

        # print(guidance_points_set) 
        #We will obtain cross-class and per-class masks so that we generate cross-class fused, and per-class fused masks.

        cross_class_masks = []
        
        per_class_masks = dict() 

        for class_label in self.class_code_dict.keys():
            
            if not include_background:
                if class_label.title()=="Background":
                    continue 
            
            per_class_masks[class_label] = []

            per_class_click_availability_bool[class_label] = guidance_points_set[class_label] != []

        for item in self.click_weightmap_types:
            
            if item == "Ellipsoid":    
                
                #If we only consider a single fused mask, We need to flatten the guidance_point parameterisations for the given mask-level dict (i.e. the dict with class-point combinations).

                # list_of_guidance_point_parametrisations = list(chain.from_iterable(list(guidance_point_parametrisations[item].values())))

                # mask = self.generate_ellipsoids(list_of_points, list_of_guidance_point_parametrisations, image_dims)


                '''
                We would like to generate a cross-class fused, and a per class fused set of masks.

                For each class, the masks are pre-fused across intra-class points. Similarly for the cross-class one across all point masks.

                '''

                cross_class_mask, per_class_mask = self.generate_ellipsoids(guidance_points_set,guidance_point_parametrisations[item], include_background, image_dims, click_availability_bool, per_class_click_availability_bool)


                cross_class_masks.append(cross_class_mask)

                for key,val in per_class_mask.items():
                    per_class_masks[key].append(val) 

            elif item == "Cuboid":
                
                #If we only consider a single fused mask, We need to flatten the guidance_point parameterisations for the given mask-level dict (i.e. the dict with class-point combinations).
                # list_of_guidance_point_parametrisations = list(chain.from_iterable(list(guidance_point_parametrisations[item].values())))

                # mask = self.generate_cuboids(list_of_points, item[1], mask)
                # masks.append(mask)

                '''
                We would like to generate a cross-class fused, and a per class fused set of masks.

                For each class, the masks are pre-fused across intra-class points. Similarly for the cross-class one across all point masks.

                '''
                
                cross_class_mask, per_class_mask = self.generate_cuboids(guidance_points_set, guidance_point_parametrisations[item], include_background, image_dims, click_availability_bool, per_class_click_availability_bool)

                cross_class_masks.append(cross_class_mask)

                for key,val in per_class_mask.items():
                    per_class_masks[key].append(val)
                


            elif item == "Scaled Euclidean Distance":
                raise NotImplementedError('Not efficient')
                #A scaled euclidean-distance that is scaled in each dimension. 

                #For scale factor = 1, this reduces to the standard euclidean distance.

                #IF we only consider a single fused mask, We need to flatten the guidance_point parameterisations for the given mask-level dict (i.e. the dict with class-point combinations).
                
                # list_of_guidance_point_parametrisations = list(chain.from_iterable(list(guidance_point_parametrisations[item].values())))

                '''
                We would like to generate a cross-class fused, and a per class fused set of masks.

                For each class, the masks are generated across intra-class points, then fused into one mask. Similarly for the cross-class one, but across all point masks.

                '''
                
                output_maps = self.generate_euclideans(True, guidance_point_parametrisations[item], guidance_points_set, include_background, image_dims, click_availability_bool, per_class_click_availability_bool, True)  
                
                assert type(output_maps) == dict 

                #Output maps are to be a dict with keys = classes, vals = lists of click distance maps.

                fusion_strategy = "Additive"

                for key, map_list in output_maps.items():
                    per_class_masks[key].append(self.map_fusion(fusion_strategy, map_list)) 
                 
                #We flatten all of the point masks into a list and fuse into one mask for the cross-clask mask.
                flattened = list(chain.from_iterable(list(output_maps.values() )))
                cross_class_masks.append(self.map_fusion(fusion_strategy, [i for i in flattened if not i.isnan().any()])) #we remove any of the nan tensors for cross class mask computation, WE ASSUME AT LEAST ONE CLICK ACROSS CLASSES! 

                # masks.append(mask)
                

            elif item == "Exponentialised Scaled Euclidean Distance":
               
                raise NotImplementedError('Not efficient')

                #IF We only consider a single fused mask, need to flatten the guidance_point parameterisations for the given mask-level dict (i.e. the dict with class-point combinations).
                
                # list_of_guidance_point_parametrisations_dummy = list(chain.from_iterable(list(guidance_point_parametrisations[item].values())))
                
                '''
                This process generates a gaussian map for each point.

                We would like to generate a cross-class fused, and a per class fused set of masks.

                For each class, the masks are generated across intra-class points, exponentiated, then fused into one mask. Similarly for the cross-class one, but across all point masks.

                
                #We assume that the parametrisation also contains the information about the exponentiation parameter. And that it is the last parameter for each point's parametrisation.
                # 
                # Therefore it is length (n + 1) where n = the quantity of scaling parameters provided (1 or 2/3).
                
                #We assume that the exponentiation is only performed according to a single parametrisation. Any per dimension modification is to be done within the per-dimension scaling 
                #of the euclidean distance.

                '''

                # list_of_guidance_point_parametrisations = [sublist[:-1] for sublist in list_of_guidance_point_parametrisations_dummy]
                # list_of_exponentiation_parameters = [sublist[-1] for sublist in list_of_guidance_point_parametrisations_dummy]

                dict_of_scale_parametrisations = dict()
                dict_of_exponentiation_parametrisations = dict()

                for class_label, list_of_point_parametrisations in guidance_point_parametrisations[item].items():

                    dict_of_scale_parametrisations[class_label] = [sublist[:-1] for sublist in list_of_point_parametrisations]
                    dict_of_exponentiation_parametrisations[class_label] = [sublist[-1] for sublist in list_of_point_parametrisations]

                #We set sqrt_bool = False here because we want to compute a gaussian weightmap.
                output_maps = self.generate_euclideans(True, dict_of_scale_parametrisations, guidance_points_set, include_background, image_dims, click_availability_bool, per_class_click_availability_bool, False)  

                assert type(output_maps) == dict
                #Output maps is a class-separated dict containing  lists of point separated masks.
                
                output_maps = self.exponentiate_map(dict_of_exponentiation_parametrisations, include_background, output_maps)

                assert type(output_maps) == dict
                #Output maps is a class-separated dict containing lists of point separated masks. 

                # print(guidance_points_set)
                fusion_strategy = "Additive"

                for key, map_list in output_maps.items():

                    per_class_masks[key].append(self.map_fusion(fusion_strategy, map_list)) #We fuse the intra-class point masks.

                #We flatten all of the point masks into a list and fuse into one mask for the cross-clask mask.
                flattened = list(chain.from_iterable(list(output_maps.values() )))
                cross_class_masks.append(self.map_fusion(fusion_strategy, [i for i in flattened if not i.isnan().any()])) #we remove any of the nan tensors for cross class mask computation, WE ASSUME AT LEAST ONE CLICK ACROSS CLASSES! 

            elif item == "Binarised Exponentialised Scaled Euclidean Distance":
               
                raise NotImplementedError('This approach will not work here efficiently')
                #IF We only consider a single fused mask, need to flatten the guidance_point parameterisations for the given mask-level dict (i.e. the dict with class-point combinations).
                
                # list_of_guidance_point_parametrisations_dummy = list(chain.from_iterable(list(guidance_point_parametrisations[item].values())))
                
                '''
                We would like to generate a cross-class fused, and a per class fused set of masks.

                For each class, the masks are generated across intra-class points, exponentiated, then fused into one mask. Similarly for the cross-class one, but across all point masks.

                
                #We assume that the parametrisation also contains the information about the exponentiation parameter. 
                # And that it is the second to last parameter for each point's parametrisation.

                We also assume that a binarisation parameter is provided for binarising the probabilistic map.
                
                # Therefore it is length (n + 2) where n = the quantity of scaling parameters provided (1 or 2/3).
                
                #We assume that the exponentiation is only performed according to a single parametrisation. Any per dimension modification is to be done within the per-dimension scaling 
                #of the euclidean distance.

                '''

                # list_of_guidance_point_parametrisations = [sublist[:-1] for sublist in list_of_guidance_point_parametrisations_dummy]
                # list_of_exponentiation_parameters = [sublist[-1] for sublist in list_of_guidance_point_parametrisations_dummy]

                dict_of_scale_parametrisations = dict()
                dict_of_exponentiation_parametrisations = dict()
                binarisation_parameter = None 

                for class_label, list_of_point_parametrisations in guidance_point_parametrisations[item].items():

                    dict_of_scale_parametrisations[class_label] = [sublist[:-2] for sublist in list_of_point_parametrisations]
                    dict_of_exponentiation_parametrisations[class_label] = [sublist[-2] for sublist in list_of_point_parametrisations]

                    if binarisation_parameter == None:
                        
                        binarisation_parameter = list_of_point_parametrisations[0][-1]
                        for sublist in list_of_point_parametrisations[1:]:
                            assert sublist[-1] == binarisation_parameter
                        
                    else:

                        for sublist in list_of_point_parametrisations:
                            assert sublist[-1] == binarisation_parameter

                output_maps = self.generate_euclideans(True, dict_of_scale_parametrisations, guidance_points_set, include_background, image_dims, click_availability_bool, per_class_click_availability_bool, False)  

                assert type(output_maps) == dict
                #Output maps is a class-separated dict containing  lists of point separated masks.
                
                output_maps = self.exponentiate_map(dict_of_exponentiation_parametrisations, include_background, output_maps)

                assert type(output_maps) == dict
                #Output maps is a class-separated dict containing lists of point separated masks. 

                # print(guidance_points_set)
                fusion_strategy = "Additive"

                for key, map_list in output_maps.items():
                    
                    #We binarise this probabilistic map according to the binarisation parameter (i.e. the probabilistic value for binarisation)
                    binarised_fused_map = torch.where(self.map_fusion(fusion_strategy, map_list) > binarisation_parameter,1 , 0)
                    per_class_masks[key].append(binarised_fused_map) #We fuse the intra-class point masks.

                #We flatten all of the point masks into a list and fuse into one mask for the cross-clask mask.
                flattened = list(chain.from_iterable(list(output_maps.values() )))
                fused_map = self.map_fusion(fusion_strategy, [i for i in flattened if not i.isnan().any()])
                #we remove any of the nan tensors for cross class mask computation, WE ASSUME AT LEAST ONE CLICK ACROSS CLASSES!

                cross_class_masks.append(torch.where(fused_map > binarisation_parameter, 1, 0)) #We append the binarised map  
            
            elif item == "2D Intersections":
                
                '''
                These intersections are pre-fused intra-class according to the set of points.
                '''
                output_maps = self.generate_axial_intersections(guidance_points_set, include_background, image_dims, click_availability_bool, per_class_click_availability_bool)
                
                assert type(output_maps) == dict 

                #Output maps is a dict containing class separated point separated masks. 
                
                for key, map in output_maps.items():

                    per_class_masks[key].append(map)
                
                #We use a union of the bools since it should capture all of the possible axial slices corresponding to the points. 

                fusion_strategy = 'Union'
                #We flatten all of the point masks into a list and fuse into one mask for the cross-clask mask.
                flattened = list(chain.from_iterable(list(output_maps.values() )))
                cross_class_masks.append(self.map_fusion(fusion_strategy, [i for i in flattened if not i.isnan().any()])) #we remove any of the nan tensors for cross class mask computation, WE ASSUME AT LEAST ONE CLICK ACROSS CLASSES! 


                # masks.append(mask)

            elif item == "None":
                mask = torch.ones(image_dims).to(device=self.device)

                cross_class_masks.append(mask)
                
                for key in self.class_code_dict.keys():
                    if not include_background:
                        if key.title() == "Background":
                            continue 
                    
                    per_class_masks[key].append(mask)
            

        
        #We then fuse across the cross-class list of masks, and the intra-class lists of masks.

        cross_class_fused = self.map_fusion('Multiplicative',cross_class_masks)
        per_class_fused = dict()

        for class_label, sublist in per_class_masks.items():
            per_class_fused[class_label] = self.map_fusion('Multiplicative', sublist)

        assert type(cross_class_fused) == torch.Tensor
        assert type(per_class_fused) == dict 

        return (cross_class_fused, per_class_fused) 
        
    def gt_based_weightmaps(self, guidance_points_set, include_background, image_dims, gt):
        
        assert type(guidance_points_set) == dict, "The generation of gt_based weightmap failed because the guidance points provided were not in class-based dict format"
        # assert type(class_code_dict) == dict, "The generation of gt_based weightmap failed because the class label - class code mapping was not provided in a dict format"
        assert type(include_background) == bool, "The generation of gt_based weightmap failed because the include_background parameter was not a bool"
        assert type(image_dims) == torch.Size, "The generation of gt_based weightmap failed because the image dimensions were not provided in torch.Size datatype"
        assert type(gt) == torch.Tensor or gt == None, "The generation of the gt_based weightmap failed because the ground truth provided was not a torch tensor or nonetype"
        assert gt.size() == image_dims, "The image_dims did not match the ground truth size"
        
        click_availability_bool = list(chain.from_iterable(list(guidance_points_set.values()))) != [] #Checking that the overall click set isn't empty
        per_class_click_availability_bool = dict()    #Checking whether each click class is empty or not.

        #We will obtain cross-class and per-class masks. 


        cross_class_masks = []
        per_class_masks = dict() 

        for class_label in self.class_code_dict.keys():
            if not include_background:
                if class_label.title()=="Background":
                    continue 
            per_class_masks[class_label] = []

            per_class_click_availability_bool[class_label] = guidance_points_set[class_label] != []


        for item in self.gt_weightmap_types:
            
            if item == "Connected Component":
                
                cross_class_mask, class_separated_masks, connected_components_dict = self.generate_gt_click_components(include_background, guidance_points_set, gt, self.class_code_dict, image_dims, click_availability_bool, per_class_click_availability_bool)
                
               
                cross_class_masks.append(cross_class_mask)

                for key in class_separated_masks.keys():
                    per_class_masks[key] = class_separated_masks[key]

            elif item == "None":
                
                mask = torch.ones(image_dims).to(device=self.device)

                cross_class_masks.append(mask)
                
                for key in self.class_code_dict.keys():
                    if not include_background:
                        if key.title() == "Background":
                            continue 
                        
                    per_class_masks[key].append(mask)
                

        #We then fuse across the cross-class list of masks, and the intra-class lists of masks.

        cross_class_fused = self.map_fusion('Multiplicative', cross_class_masks)
        per_class_fused = dict()

        for class_label, sublist in per_class_masks.items():
            per_class_fused[class_label] = self.map_fusion('Multiplicative', sublist)

        assert type(cross_class_fused) == torch.Tensor
        assert type(per_class_fused) == dict 

        return cross_class_fused, per_class_fused  
    
    def human_measure_weightmap(self, click_based_weightmaps, gt_based_weightmaps, human_measure_information, image_dims):
        '''        
        The click based weightmaps are a tuple: cross_class_fused weightmap which is fused across all of the selected click-based types, and the per-class ones fused across the selected click-based types
        Same for the gt based weightmap
        
        The human_measure_information is any information required for the human_measure weightmap. In the case of Temporal Non Worsening for example, this would be the set of changed voxels, and the class-split set.
        '''

        assert type(click_based_weightmaps) == tuple  
        assert type(gt_based_weightmaps) == tuple 

        assert type(click_based_weightmaps[0]) == torch.Tensor 
        assert type(click_based_weightmaps[1]) == dict 

        assert type(gt_based_weightmaps[0]) == torch.Tensor
        assert type(gt_based_weightmaps[1]) == dict

        assert type(human_measure_information) == dict or human_measure_information == None 

        

        if self.human_measure[0].title() == "Local Responsiveness":
            #If locality or temporal consistency is the measure then no change required. We just need to fuse the click based and gt based weightmaps

            cross_class_click_weightmap = click_based_weightmaps[0]
            cross_class_gt_weightmap = gt_based_weightmaps[0] 

            final_cross_class_weightmap = cross_class_click_weightmap * cross_class_gt_weightmap 
            
            per_class_final_weightmaps = dict() 
            
            for class_label in click_based_weightmaps[1].keys():
                
                per_class_click_weightmap = click_based_weightmaps[1][class_label]
                per_class_gt_weightmap = gt_based_weightmaps[1][class_label]

                final_weightmap = per_class_click_weightmap * per_class_gt_weightmap 

                per_class_final_weightmaps[class_label] = final_weightmap 
            
        



        elif self.human_measure[0].title() == "Temporal Non Worsening":
            #If temporal non-worsening is the measure, then we need to invert a locality based mask.
            
            #For temporal non-worsening, we also have the set of changed voxels for that class (and include background) as a bool mask.
            
            cross_class_information = human_measure_information['cross_class_changed_voxels']
            cross_class_click_weightmap = click_based_weightmaps[0]
            cross_class_gt_weightmap = gt_based_weightmaps[0] 


            final_cross_class_weightmap = (1 - cross_class_click_weightmap * cross_class_gt_weightmap) * cross_class_information

            per_class_final_weightmaps = dict() 
            
            #We use the class labels in the human measure information dict, because we may want information about background clicks BUT there may be an instance where
            #we do not want any information about the background class for the metric computation -- the human measure information is used to pass forward this information
            #implicitly. 

            for class_label in human_measure_information['per_class_changed_voxels'].keys():
                
                per_class_click_weightmap = click_based_weightmaps[1][class_label]
                per_class_gt_weightmap = gt_based_weightmaps[1][class_label]
                per_class_information = human_measure_information['per_class_changed_voxels'][class_label]

                final_weightmap = (1 - per_class_click_weightmap * per_class_gt_weightmap) * per_class_information

                per_class_final_weightmaps[class_label] = final_weightmap  

        elif self.human_measure[0].title() == "Temporal Consistency":

            #If temporal non-worsening is the measure, then we need to invert a locality based mask.
            
            #For temporal consistency, we DO NOT have the set of changed voxels for that class (and include background) as a bool mask.

            cross_class_click_weightmap = click_based_weightmaps[0]
            cross_class_gt_weightmap = gt_based_weightmaps[0] 


            final_cross_class_weightmap = (1 - cross_class_click_weightmap * cross_class_gt_weightmap)

            per_class_final_weightmaps = dict() 
            
            #We use the class labels in the human measure information dict, because we may want information about background clicks BUT there may be an instance where
            #we do not want any information about the background class for the metric computation -- the human measure information is used to pass forward this information
            #implicitly. 

            for class_label in click_based_weightmaps[1].keys():
                
                per_class_click_weightmap = click_based_weightmaps[1][class_label]
                per_class_gt_weightmap = gt_based_weightmaps[1][class_label]
            
                final_weightmap = (1 - per_class_click_weightmap * per_class_gt_weightmap)

                per_class_final_weightmaps[class_label] = final_weightmap  

        elif self.human_measure[0].title() == "None":
            #This is just for default scores! No weightmap
            final_cross_class_weightmap = torch.ones(image_dims).to(device=self.device)
            
            per_class_final_weightmaps = dict() 
            
            for class_label in click_based_weightmaps[1].keys():
                
                per_class_final_weightmaps[class_label] = torch.ones(image_dims).to(device=self.device)
            

                

        assert type(final_cross_class_weightmap) == torch.Tensor 
        assert type(per_class_final_weightmaps) == dict 
        
        return final_cross_class_weightmap, per_class_final_weightmaps
    


    def __call__(self, guidance_points_set, point_parametrisations, include_background, human_measure_information, image_dims, gt):
        

        '''
        In instances where a weightmap subtype is not being used (e.g. click or gt type), the "None" will be the only corresponding selection in that list for the mask generator definition.

        Therefore, in these instances it will just generate tensors of ones.

        Per Class masks are generated using the clicks/gt information ONLY for that class. Cross class masks are generated across all classes (including background if specified)
        
        '''
        
        '''This call functions accepts the set of guidance points, and a corresponding set of parameterisations for the selected mask types AND corresponding to the given click.
        It also accepts information about the human_measure, for example with temporal non-worsening it would be the map of changed voxels. It also takes information about whether the
        background is included.
        

        #This takes the dictionary which contains the guidance points for all classes. This set of guidance points will be assumed to be in the same orientation of the images.'''
        
        
        assert type(guidance_points_set) == dict
        
        '''Ensure that the value for each class key (class name) in the dict is a nested list of the points which are represented as a list of coordinates'''

        for guidance_point_list in guidance_points_set.values():

            if any([type(entry) != list for entry in guidance_point_list]):
                raise Exception("Non-list entry in the list of guidance points")

        '''Also takes a dictionary which contains the parameterisations for each mask type selected AND for each point.
        
        #The structure is therefore a nested dictionary, the upper level is for the mask types, and the nested level is for the set of guidance points. Each class of guidance points has a 
        #nested list containing the parameterisation for that MASK TYPE AND CLICK.'''

        #Asserting dictionary at the upper level
        assert type(point_parametrisations) == dict, "Input point parametrisation was not a dictionary"

        #Asserting that each value must also be a dictionary.
        for value_entry in point_parametrisations.values():
            assert type(value_entry) == dict, "Mask-level structure in point parameterisation was not a dictionary"

            #Asserting that each value in the dictionary must be a nested list for each segmentation class.  
            for class_point_parametrisations in value_entry.values():
                assert type(class_point_parametrisations) == list, "Class-level structure in point parametrisation was not a list"

                for point_level_parametrisation in class_point_parametrisations:
                    assert type(point_level_parametrisation) == list, "Point-level structure in point parametrisation was not a list"
        

        ''' Also takes a list containing the dimensions of the image, for which the weight-maps will be created, assumed to be in RAS orientation for 3D images'''

        assert type(image_dims) == torch.Size, "Image dimensions were not in a torch.Size datatype"
        
        assert type(human_measure_information) == dict or human_measure_information == None, "Human information was not in the appropriate datatype"
        
        assert type(include_background) == bool, "Information about including the background was not provided in the bool format."

        assert type(gt) == torch.Tensor or gt == None, "The ground truth provided was not in the right format, torch.Tensor or NoneType"
        
        
        
        cross_class_click_weightmaps, per_class_click_weightmaps = self.click_based_weightmaps(guidance_points_set, point_parametrisations, include_background, image_dims)

        click_weightmaps = (cross_class_click_weightmaps, per_class_click_weightmaps)
            
            
        cross_class_gt_weightmaps, per_class_gt_weightmaps = self.gt_based_weightmaps(guidance_points_set, include_background, image_dims, gt)

        gt_weightmaps = (cross_class_gt_weightmaps, per_class_gt_weightmaps)

        cross_class_map, per_class_maps = self.human_measure_weightmap(click_weightmaps, gt_weightmaps, human_measure_information, image_dims)
        
        assert type(cross_class_map) == torch.Tensor 
        assert type(per_class_maps) == dict 
         
        return cross_class_map, per_class_maps 


    def generate_gt_click_components(self, include_background, guidance_points_set, gt, class_labels, image_dims, click_avail_bool, per_class_avail_bool):
        
        assert type(include_background) == bool, "Generation of connected_component map failed because the include_background parameter was not a bool"
        assert type(guidance_points_set) == dict, "Generation of connected component containing the click failed due to the points not being provided in a class-separated dict"
        assert type(image_dims) == torch.Size, "The generation of connected_component map failed because the image dimensions were not provided in torch.Size datatype"
        assert type(class_labels) == dict, "The generation of connected_component map failed because the class labels provided were not in a dict"
        assert type(gt) == torch.Tensor, "The generation of connected_component map failed because the ground truth provided was not a torch tensor"
        assert gt.size() == image_dims, "The image_dims did not match the ground truth size" 
        assert click_avail_bool == True, "The generation of connected component map failed because the parameter that determined whether the cross class click set was available was false"
        assert type(per_class_avail_bool) == dict, "The generation of connected component map failed because the param that contained the per class click avail bools was not a dict"

        #The class labels are the integer codes that correspond to the predicted segmentation AND ground truth segmentation label codes for each class.
        #We assume that the ground truth is fully discrete also.

        #First we extract the connected components for each class in the ground truth, and the number of components for each of those also.
        
        raise ValueError('We have not yet debugged this mask generator type')


        connected_components_dict = dict() 

        class_separated_masks = dict() 

        for class_label, class_code in class_labels.items():
            assert type(class_code) == int, "Generation of connected components containing the clicks failed due to the class code not being an int"

            if not include_background: 
                if class_label.title() == 'Background':
                    continue 
            connected_components_dict[class_label] = self.connected_component_generator(gt, class_code)


            #If include_background = False then any click in a background will be skipped


            #We will find the component for each corresponding click, and then combine them all into one boolean mask.
            
            #Extracting the point's corresponding connected component.
            
            point_components = [] 

            for point in guidance_points_set[class_label]:
            
                if len(image_dims) == 2:
                   
                    point_component_label = connected_components_dict[class_label][0][point[0], point[1]]

                elif len(image_dims) == 3:
                   
                    point_component_label = connected_components_dict[class_label][0][point[0], point[1], point[2]]

                point_component = torch.where(connected_components_dict[class_label][0] == point_component_label, 1, 0) 
                
                point_components.append(point_component)
            
            if per_class_avail_bool[class_label] == False: #In this case clicks were not available for this class
            
                point_components = [torch.nan * torch.ones(image_dims)]

            class_separated_masks[class_label] = self.map_fusion('Union', point_components)
        

        assert type(class_separated_masks) == dict, "The output class separated masks for the click component function were not contained in a dict"
        
        for val in class_separated_masks.values():
            assert type(val) == torch.Tensor, "The class separated masks in the click component function were not torch tensors"

        fused_mask = self.map_fusion('Union', list(class_separated_masks.values()))            

        assert type(fused_mask) == torch.Tensor, "The output fused mask for the click component mask was not a torch tensor"

        return fused_mask, class_separated_masks, connected_components_dict  
    


    def connected_component_generator(self, gt, class_code):
        raise NotImplementedError('This needs to be modified for efficiency')
        bool_mask = torch.where(gt == class_code, 1, 0)


        #First we split the ground truth map into the disconnected components.
        disconnected_components_map, num_components = connected_comp_measure.label(bool_mask, return_num=True, connectivity=len(gt.size()))

        output_map = torch.from_numpy(disconnected_components_map)
        
        assert type(output_map) == torch.Tensor, "Output from the connected component generator was not a torch tensor"

        #Returns a list containing the new labelled map, and corresponding number of components.
        return [output_map, num_components]
    



    def generate_axial_intersections(self, guidance_points_set, include_background, image_dims, click_avail_bool, per_class_click_avail_bool):
        # Non-parametric, only depends on the axial slices which could've been used to place the click on a 2D monitor. This is only applicable for 3D images!
        raise NotImplementedError('Not currently supported for use in computing metrics')
        assert type(guidance_points_set) == dict, 'Generation of axial slice intersections failed due to the guidance points not being in a dict datatype'
        assert type(include_background) == bool, 'Generation of axial slice intersections failed due to the include_ background parameter not being a bool'
        assert type(image_dims) == torch.Size, 'Generation of axial slice intersections failed due to the image dimensions not being provided in a torch.Size datatype'
        assert click_avail_bool == True, 'Generation of axial slice intersections failed because it was a click was not available across classes.'
        assert type(per_class_click_avail_bool) == dict 

        #Checking that the image dims are indeed 3 dimensional.

        if len(image_dims) < 3:
            raise Exception("Selected 2D Intersections for a 2D image!") 
        
        assert type(guidance_points_set) == dict, "The parameter containing the points was not a class-separated dict"
        
        per_class_masks = dict() 

        for class_label, list_of_points in guidance_points_set.items(): 
            
            if not include_background:
                if class_label.title() == "Background":
                    continue 
            
            mask = torch.zeros(self.image_dims).to(device=self.device)
            #We initialise a tensor of zeroes, then we will insert 1s at the axial slices in which clicks could have feasibly been placed! 

            for point in list_of_points:
                mask[point[0],:,:] = 1
                mask[:,point[1],:] = 1
                mask[:,:,point[2]] = 1
            
            if per_class_click_avail_bool[class_label] == False:
                mask = torch.nan * torch.ones(image_dims).to(device=self.device)
            
            per_class_masks[class_label] = mask

        return per_class_masks 
    
    def map_fusion(self, fusion_strategy, maps):
        '''
        Map fusion function which fuses together a LIST of maps either by pure additive fusion, elementwise multiplication, or by finding the union of booleans
        '''
        supported_fusions = ["Additive", "Multiplicative", "Union"]
        
        assert fusion_strategy.title() in supported_fusions, "Selected fusion strategy is not supported by the image map fusion function"

        if  fusion_strategy == "Additive":
            summed_output_maps = sum(maps)
            
            return summed_output_maps/torch.max(summed_output_maps) 
        
        if fusion_strategy == "Multiplicative":

            product_output_maps = functools.reduce(operator.mul, maps, 1)
            return product_output_maps/torch.max(product_output_maps)
        
        if  fusion_strategy == "Union":

            union_output_maps = sum(maps)
            return torch.where(union_output_maps > 0, 1, 0).to(device=self.device)

    def exponentiate_map(self, dict_of_exponentiation_parameters, include_background, maps):
        '''
        Returns class-separated dict of lists of point-masks.
        '''
        output_maps = dict()

        for class_label, parametrisation in dict_of_exponentiation_parameters.items():
            
            if not include_background:
                if class_label.title() == "Background":
                    continue 
        
            class_maps = maps[class_label]

            # print(class_maps)
            # print(class_label)
            output_maps[class_label] = [torch.exp(-parametrisation[i] * weight_map).to(device=self.device) for i,weight_map in enumerate(class_maps)]
        
        return output_maps

    def generate_euclideans(self, is_normalised_bool, scaling_parametrisations_set, guidance_points_set, include_background, image_dims, click_avail_bool, per_class_click_avail_bool, square_root_bool):
        raise NotImplementedError('Not efficient strategy for large quantities of points')
    
        '''Is_normalised parameter just assesses whether the distances are scaled by the scaling parametrisations
           Axis scaling parametrisation is the scaling denominator of the summative terms of the euclidean computation.
           square_Root_bool just controls whether we square root the terms in the euclidean distance or not.
        '''
        assert type(is_normalised_bool) == bool, "Is_normalised bool parameter in euclidean map generation was not a bool"
        # assert type(additive_fusion_bool) == bool, "Additive Fusion bool parameter in euclidean map generation was not a bool"
        assert type(guidance_points_set) == dict, "Generation of euclidean map failed because points were not in a class-separated dict"
        assert type(image_dims) == torch.Size, "Generation of euclidean map failed because the image dimension provided was not torch.Size datatype"
        assert type(scaling_parametrisations_set) == dict, "Generation of euclidean map failed because the axis scaling parametrisation was not a within a class-separated dict"
        assert click_avail_bool == True 
        assert type(per_class_click_avail_bool) == dict 

        per_class_masks = dict() 
        # full_set_of_masks = []
        
        for class_label, list_of_points in guidance_points_set.items():

            if not include_background:
                if class_label.title() == "Background":
                    continue 
            
            list_of_scaling_parametrisation = scaling_parametrisations_set[class_label]

            centres = [[coord + 0.5 for coord in centre] for centre in list_of_points]

            
            intra_class_masks = []

            for i, centre in enumerate(centres):
                
                assert type(list_of_scaling_parametrisation[i]) == list, "Generation of euclidean map failed because the axis scaling parametrisation for each point was not a list"

                intra_class_masks.append(self.each_euclidean(is_normalised_bool, list_of_scaling_parametrisation[i], centre, image_dims, square_root_bool))

            if per_class_click_avail_bool[class_label] == False: #Click set was not available for this class

                intra_class_masks = [torch.ones(image_dims) * torch.nan]

            per_class_masks[class_label] = intra_class_masks

            # full_set_of_masks += intra_class_masks
            


        # assert type(cross_class_mask) == torch.Tensor 
        assert type(per_class_masks) == dict 

        return per_class_masks
    
    def each_euclidean(self, is_normalised, scaling_parametrisation, point, image_dims, square_root_bool):
        raise NotImplementedError('Not efficient for large quantities of points')
    
        '''Is_normalised parameter just assesses whether the distances are scaled by a scaling parametrisation
        Square root bool just assesses whether we square root the map or not'''
        assert type(is_normalised) == bool, "Is_normalised parameter in euclidean map generation was not a bool"
        assert type(point) == list, "Generation of euclidean map failed because point was not a list"
        assert type(image_dims) == torch.Size, "Generation of euclidean map failed because the image dimension provided was not torch.Size datatype"
        assert type(scaling_parametrisation) == list, "Scaling parametrisation for the denom terms of the euclidean were not provided in the list format"


        if len(scaling_parametrisation) == 1:
            scaling_parametrisation*= len(image_dims)
        else:
            pass

        grids = [torch.linspace(0.5, image_dim-0.5, image_dim) for image_dim in image_dims]
        meshgrid = torch.meshgrid(grids, indexing='ij')

        if square_root_bool:
            if is_normalised:
                return torch.sqrt(sum([torch.square((meshgrid[i] - point[i])/(scaling_parametrisation[i])) for i, image_dim in enumerate(image_dims)]))
            else:
                return torch.sqrt(sum([torch.square(meshgrid[i] - point[i]) for i in range(len(image_dims))]))
        
        else:
            if is_normalised:
                return sum([torch.square((meshgrid[i] - point[i])/(scaling_parametrisation[i])) for i, image_dim in enumerate(image_dims)])
            else:
                return sum([torch.square(meshgrid[i] - point[i]) for i in range(len(image_dims))])
        

    def generate_cuboids(self, guidance_points_set, scale_parametrisation_set, include_background, image_dims, click_avail_bool, per_class_click_avail_bool):
        '''
        Cuboids require parameterisation.

        Parametrisation is a set of raw parameters for each point.

        This parametrisation is the raw quantity of voxels..(e.g. 50 voxels in x, 75 in y, 90 in z) because we might have variations in the actual physical measurement per voxel (e.g. 1 x 10 x 10mm)
        
        Returns:

        Cross-class fused mask and a dict of per-class fused masks across the guidance points correspondingly.
        '''
        
        assert type(scale_parametrisation_set) == dict, "Structure of scale parametrisations across classes in cuboid generator was not a dict (with nested lists)"
        assert type(guidance_points_set) == dict, "Structure of guidance point sets across classes in cuboid generator was not a dict (with nested lists)"
        assert type(include_background) == bool 
        assert type(image_dims) == torch.Size, "Datatype for the image dimensions in cuboid generator was not torch.Size"
        assert click_avail_bool == True 
        assert type(per_class_click_avail_bool) == dict 

        for list_of_scale_parametrisation in scale_parametrisation_set.values():
            assert type(list_of_scale_parametrisation) == list, "Structure of scale parametrisations for a given class in cuboid generator was not a list"

            for sublist in list_of_scale_parametrisation:

                assert type(sublist) == list, "Structure of scale parametrisations for each point in cuboid generator was not a list"

        


        #For each class, we generate the per-class fused mask. 
        per_class_masks = dict()

        for class_label, list_of_points in guidance_points_set.items():
            
            if not include_background:
                if class_label.title() == "Background":
                    continue 
            
            #Initialising the mask:
            mask = torch.zeros(image_dims).to(device=self.device)


            #Extracting the list of scale parametrisations for the set of points of the given class. 

            list_of_scale_parametrisation = scale_parametrisation_set[class_label]


            for point, scale_parametrisation in zip(list_of_points, list_of_scale_parametrisation):

                #shifting the centre to the center of each voxel (where we assume click placement occurs)
                centre = [coord + 0.5 for coord in point]


                '''
                None of the scale parameterisations should be larger than the 0.5 of the corresponding image dimensions otherwise the box would be larger than the image.
                '''


                if len(scale_parametrisation) == 1:
                    parametrisation = scale_parametrisation * len(image_dims)
                else:
                    parametrisation = scale_parametrisation

                if any(torch.tensor(parametrisation)/torch.tensor(image_dims) > 0.5):
                    raise Exception("Scale factors for the cuboid size mean that the dimensions would be larger than the image")
                
                
                #obtain the extreme points of the cuboid which will be assigned as the box region:
                min_maxes = []


                for index, coordinate in enumerate(centre):
                    #For each coordinate, we obtain the extrema points.
                    dimension_min = int(max(0, torch.round(torch.tensor(coordinate - parametrisation[index]))))
                    dimension_max = int(min(image_dims[index] - 1, torch.round(torch.tensor(coordinate + parametrisation[index]))))

                    min_max = [dimension_min, dimension_max] 
                    min_maxes.append(min_max)


                if len(image_dims) == 2:
                #If image is 2D            
                    mask[min_maxes[0][0]:min_maxes[0][1], min_maxes[1][0]:min_maxes[1][1]] = 1
                elif len(image_dims) == 3:
                    #If image is 3D:
                    mask[min_maxes[0][0]:min_maxes[0][1], min_maxes[1][0]:min_maxes[1][1], min_maxes[2][0]:min_maxes[2][1]] = 1

            if per_class_click_avail_bool[class_label] == False: #In the scenario where the click set is completely empty for this class. 

                mask = torch.ones(image_dims).to(device=self.device) * torch.nan 

            
            #Appending the "fused" mask into the per-class mask dict.
            per_class_masks[class_label] = mask 

        #Fusing the per-class masks into a single cross class mask also.
        fusion_strategy = "Union"

        #We placed the per-class masks into a list and fuse into one mask for the cross-clask mask.
        flattened = list(per_class_masks.values() )
        cross_class_mask = self.map_fusion(fusion_strategy, [i for i in flattened if not i.isnan().any()]) #we remove any of the nan tensors for cross class mask computation, WE ASSUME AT LEAST ONE CLICK ACROSS CLASSES! 


        assert type(cross_class_mask) == torch.Tensor 
        assert type(per_class_masks) == dict 

        return cross_class_mask, per_class_masks


    def generate_ellipsoids(self, guidance_points_set, scale_parametrisations_set, include_background, image_dims, click_avail_bool, per_class_click_avail_bool):
        
        '''
        #Ellipsoid requires parametrisation: There are three options available
        

        #For each point, the following parametrisation configurations are permitted: 

        #param_1 only: All dimensions have the same scaling

        #param_1/2 or param_1/2/3 indicate separate scalings

        #In instances where it has separate scalings, this is assumed to be a list with length > 1! 

        #All parameters must be smaller than the resolution of the corresponding dimensions.

        #Mask is a torch tensor.


        Inputs: Guidance points sets, split by class. Scale parametrisations set for the guidance points. Whether the background is included: Bool. Image dimensions in the same orientation of the 
        guidance points.

        Returns:

        Fused mask of ellipsoids across the classes, and a dict of per_class ellipsoid masks corresponding to the guidance points that were provided.
        '''
    
        assert type(image_dims) == torch.Size, "Image dimensions for the ellipsoid mask generators were not of the torch.Size type"
        assert type(scale_parametrisations_set) == dict, "scale parametrisation for the ellipsoid mask generators were not of the dict datatype"
        assert type(include_background) == bool
        assert type(image_dims) == torch.Size 
        assert click_avail_bool == True
        assert type(per_class_click_avail_bool) == dict 

        per_class_masks = dict() 
        fusion_strategy = 'Union'

        for class_label, list_of_points in guidance_points_set.items():
            
            if not include_background:
                if class_label.title() == "Background":
                    continue 

            #Generate the per-class fused mask

            list_of_scale_parametrisation = scale_parametrisations_set[class_label] 
            
            assert type(list_of_scale_parametrisation) == list, "List of scale parametrisations in the ellipsoid generator was not a nested list for each class"

            #Initialise ellipsoid mask
            ellipsoids_mask = torch.zeros((image_dims)).to(device=self.device)

            for point, scale_parametrisation in zip(list_of_points, list_of_scale_parametrisation):
                
                if len(scale_parametrisation) == 1:
                    parametrisation = scale_parametrisation * len(image_dims)
                else:
                    parametrisation = scale_parametrisation

                if any(torch.tensor(parametrisation)/torch.tensor(image_dims) > 0.5):
                    raise Exception("Scale factor too large, axis of ellipse will be larger than the image")

                assert type(scale_parametrisation) == list, "Scale parametrisation for each point for ellipsoid generation was not in a list structure"

                
                ellipsoids_mask += self.each_ellipsoid(point, parametrisation, image_dims)
                
            if per_class_click_avail_bool[class_label] == False: #If there is no click set!
                ellipsoids_mask = torch.ones(image_dims).to(device=self.device) * torch.nan
            else:
                ellipsoids_mask = torch.where(ellipsoids_mask > 0, 1, 0)

            per_class_masks[class_label] = ellipsoids_mask

        #We placed the per-class masks into a list and fuse into one mask for the cross-clask mask.
        flattened = list(per_class_masks.values() )
        cross_class_mask = self.map_fusion(fusion_strategy, [i for i in flattened if not i.isnan().any()]) #we remove any of the nan tensors for cross class mask computation, WE ASSUME AT LEAST ONE CLICK ACROSS CLASSES! 

        # cross_class_mask = self.map_fusion(fusion_strategy, list(chain.from_iterable(list(per_class_masks.values() ))))

        assert type(cross_class_mask) == torch.Tensor 
        assert type(per_class_masks) == dict 
         
        return cross_class_mask, per_class_masks


    def each_ellipsoid(self,centre, scale_factor_denoms, image_dims):
        #Generating the bool mask outlining the ellipsoid defined using the centre, and the scale parameters corresponding to each image dimension. This ranges from [0, 0.5]
        
        
        #Ellipsoids is defined in the following manner: (x-xo/a)^2 + (y-yo/b)^2 + (z-zo/c)^2 = 1 (for 2D the z-term is just dropped)

        #We treat point coordinates as being at the center of each voxel.  
        
        #We create a grid of coordinates for the image:
        grids = [torch.linspace(0.5, image_dim - 0.5, image_dim).to(device=self.device) for image_dim in image_dims]

        #shifting the centre 
        centre = [coord + 0.5 for coord in centre]

        #computing the denominators using the scale_factors for each image_dimension
        denoms =  scale_factor_denoms #[scale_factor_denoms[i] for i,image_dim in enumerate(image_dims)]
        
        #generating the coordinate set
        
        # if len(image_dims) == 2:
        #     meshgrid = torch.meshgrid(grids[0], grids[1], indexing='ij')
        # else:
        #     meshgrid = torch.meshgrid(grids[0], grids[1], grids[2], indexing='ij')   

        meshgrid = torch.meshgrid(grids, indexing='ij')
        
        
        lhs_comp = sum([torch.square((meshgrid[i] - centre[i])/denoms[i]) for i in range(len(image_dims))])

        return torch.where(lhs_comp <= 1, 1, 0).to(device=self.device) #potentially overkill but just in case.
    

if __name__ == '__main__':
    import time 

    mask_gen_class = MaskGenerator(['Cuboid'], ['None'], 'Local Responsiveness', {'tumour':1, 'background':0}, True, "cuda:0")
    gt = torch.ones((128,128,128))
    img_dims = torch.Size([128,128,128])
    human_measurement_information = None
    point_parametrisations = {'Cuboid':{'tumour':[[5,5,5] for i in range(50)], 'background':[[5,5,5]]}}
    guidance_points_set = {'tumour':[[1,2,3] for i in range(50)], 'background':[]}
    include_background = True 

    for i in range(30):
        start = time.time()
        dummy_mask = mask_gen_class(guidance_points_set, point_parametrisations, include_background, human_measurement_information, img_dims, gt)
        end = time.time()
        print(end-start)