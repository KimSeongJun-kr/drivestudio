#!/usr/bin/env python3
"""
nuScenes 데이터 분석 도구 (NuScenes API 사용)
센서 종류별로 prev, next 정보를 이용하여 데이터를 순서대로 파싱하고 scene별로 구분
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import argparse
from nuscenes.nuscenes import NuScenes


class NuScenesFrameAnalyzer:
    """NuScenes API를 사용하여 센서별 프레임 순서를 파싱하는 클래스"""
    
    def __init__(self, dataroot: str, version: str = 'v1.0-mini'):
        self.dataroot = dataroot
        self.version = version
        self.nusc = None
        self.sensor_chains = defaultdict(list)
        self.scene_sensor_chains = defaultdict(lambda: defaultdict(list))
        self.available_sensors = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT',
            'LIDAR_TOP', 'RADAR_FRONT', 'RADAR_FRONT_LEFT', 
            'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT'
        ]
        
    def load_data(self):
        """NuScenes 데이터 로드"""
        print(f"NuScenes 데이터 로딩 중...")
        print(f"  데이터 루트: {self.dataroot}")
        print(f"  버전: {self.version}")
        
        try:
            self.nusc = NuScenes(version=self.version, dataroot=self.dataroot, verbose=True)
            print(f"✓ 총 {len(self.nusc.scene)}개의 scene을 로드했습니다.")
            print(f"✓ 총 {len(self.nusc.sample)}개의 sample을 로드했습니다.")
            print(f"✓ 총 {len(self.nusc.sample_data)}개의 sample_data를 로드했습니다.")
        except Exception as e:
            raise RuntimeError(f"NuScenes 데이터 로드 실패: {e}")
    
    def get_sensor_sequence(self, scene_token: str, channel: str) -> List[Dict]:
        """특정 scene과 센서에 대한 시간순 sequence 반환"""
        try:
            scene = self.nusc.get('scene', scene_token)
            sample0 = self.nusc.get('sample', scene['first_sample_token'])
            
            # 해당 센서의 sample_data가 있는지 확인
            if channel not in sample0['data']:
                return []
            
            sd_token = sample0['data'][channel]
            sd = self.nusc.get('sample_data', sd_token)
            
            sequence = []
            idx = 0
            keyframe_idx = 0
            
            while True:
                # sample_data 정보를 sequence에 추가
                sequence_entry = {
                    'sequence_idx': idx,
                    'keyframe_idx': keyframe_idx if sd['is_key_frame'] else None,
                    'token': sd['token'],
                    'sample_token': sd['sample_token'],
                    'timestamp': sd['timestamp'],
                    'filename': sd['filename'],
                    'is_key_frame': sd['is_key_frame'],
                    'prev_token': sd['prev'],
                    'next_token': sd['next'],
                    'channel': channel,
                    'scene_token': scene_token,
                    'scene_name': scene['name']
                }
                sequence.append(sequence_entry)
                
                # 키프레임인 경우 keyframe_idx 증가
                if sd['is_key_frame']:
                    keyframe_idx += 1
                
                # 다음 sample_data로 이동
                if sd['next'] == '':
                    break
                sd = self.nusc.get('sample_data', sd['next'])
                idx += 1
                
            return sequence
            
        except Exception as e:
            print(f"  경고: {channel} 센서의 sequence를 가져올 수 없습니다: {e}")
            return []
    
    def get_keyframe_indices(self, scene_token: str, channel: str) -> List[int]:
        """키프레임(2Hz 샘플)의 인덱스 리스트 반환"""
        try:
            scene = self.nusc.get('scene', scene_token)
            sample0 = self.nusc.get('sample', scene['first_sample_token'])
            
            if channel not in sample0['data']:
                return []
                
            sd_token = sample0['data'][channel]
            sd = self.nusc.get('sample_data', sd_token)
            
            idx, key_indices = 0, []
            while True:
                if sd['is_key_frame']:
                    key_indices.append(idx)
                if sd['next'] == '':
                    break
                sd = self.nusc.get('sample_data', sd['next'])
                idx += 1
                
            return key_indices
            
        except Exception as e:
            print(f"  경고: {channel} 센서의 키프레임을 가져올 수 없습니다: {e}")
            return []
    
    def analyze_all_scenes_and_sensors(self):
        """모든 scene과 센서에 대해 분석 수행"""
        print(f"\n모든 scene과 센서 분석 중...")
        
        # 전체 센서별 체인 초기화
        for sensor in self.available_sensors:
            self.sensor_chains[sensor] = []
        
        for scene_idx, scene in enumerate(self.nusc.scene):
            scene_token = scene['token']
            scene_name = scene['name']
            
            print(f"\n--- Scene {scene_idx+1}/{len(self.nusc.scene)}: {scene_name} ---")
            print(f"  설명: {scene['description']}")
            print(f"  샘플 수: {scene['nbr_samples']}")
            
            scene_total_frames = 0
            
            for sensor in self.available_sensors:
                # 각 센서별 sequence 가져오기
                sequence = self.get_sensor_sequence(scene_token, sensor)
                
                if sequence:
                    # 전체 센서 체인에 추가
                    self.sensor_chains[sensor].extend(sequence)
                    
                    # scene별 센서 체인에 추가
                    self.scene_sensor_chains[scene_name][sensor] = sequence
                    
                    # 키프레임 인덱스 확인
                    key_indices = self.get_keyframe_indices(scene_token, sensor)
                    
                    scene_total_frames += len(sequence)
                    print(f"    {sensor:18}: {len(sequence):4}개 프레임 (키프레임: {len(key_indices)}개)")
                else:
                    print(f"    {sensor:18}: 데이터 없음")
            
            print(f"    {'Scene 총계':18}: {scene_total_frames:4}개 프레임")
    
    def export_sensor_sequences(self, output_dir: str = "output"):
        """센서별 순서대로 정렬된 데이터를 파일로 저장"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"\n센서별 순서 데이터를 {output_dir}에 저장 중...")
        
        # 전체 센서별 저장
        for sensor, chain in self.sensor_chains.items():
            if not chain:
                continue
                
            output_file = os.path.join(output_dir, f"{sensor}_sequence.json")
            
            with open(output_file, 'w') as f:
                json.dump(chain, f, indent=2)
            
            print(f"  {sensor}: {len(chain)}개 프레임 -> {output_file}")
        
        # Scene별 센서 시퀀스 저장
        if self.scene_sensor_chains:
            self.export_scene_sequences(output_dir)
    
    def export_scene_sequences(self, output_dir: str):
        """Scene별로 센서 시퀀스 데이터를 저장"""
        scenes_dir = os.path.join(output_dir, "scenes")
        if not os.path.exists(scenes_dir):
            os.makedirs(scenes_dir)
        
        print(f"\nScene별 센서 시퀀스를 {scenes_dir}에 저장 중...")
        
        for scene_name, sensor_chains in self.scene_sensor_chains.items():
            scene_dir = os.path.join(scenes_dir, scene_name)
            if not os.path.exists(scene_dir):
                os.makedirs(scene_dir)
            
            # Scene 정보 가져오기
            scene_info = None
            for scene in self.nusc.scene:
                if scene['name'] == scene_name:
                    scene_info = scene
                    break
            
            # Scene 정보 저장
            if scene_info:
                scene_info_file = os.path.join(scene_dir, "scene_info.json")
                with open(scene_info_file, 'w') as f:
                    json.dump(scene_info, f, indent=2)
            
            print(f"\n  === {scene_name} ===")
            total_frames = 0
            
            for sensor, sequence in sensor_chains.items():
                if not sequence:
                    continue
                
                output_file = os.path.join(scene_dir, f"{sensor}_sequence.json")
                
                # Scene 정보를 포함한 확장된 데이터 생성
                enhanced_sequence = []
                for entry in sequence:
                    enhanced_entry = entry.copy()
                    if scene_info:
                        enhanced_entry.update({
                            'scene_description': scene_info['description'],
                            'scene_nbr_samples': scene_info['nbr_samples'],
                            'scene_first_sample_token': scene_info['first_sample_token'],
                            'scene_last_sample_token': scene_info['last_sample_token']
                        })
                    enhanced_sequence.append(enhanced_entry)
                
                with open(output_file, 'w') as f:
                    json.dump(enhanced_sequence, f, indent=2)
                
                total_frames += len(sequence)
                print(f"    {sensor}: {len(sequence)}개 프레임")
            
            print(f"    총 {total_frames}개 프레임 -> {scene_dir}/")
        
        print(f"\n총 {len(self.scene_sensor_chains)}개 Scene으로 분리하여 저장 완료")
    
    def export_keyframe_analysis(self, output_dir: str):
        """키프레임 분석 결과를 별도로 저장"""
        keyframe_dir = os.path.join(output_dir, "keyframes")
        if not os.path.exists(keyframe_dir):
            os.makedirs(keyframe_dir)
        
        print(f"\n키프레임 분석 결과를 {keyframe_dir}에 저장 중...")
        
        for scene_name, sensor_chains in self.scene_sensor_chains.items():
            keyframe_file = os.path.join(keyframe_dir, f"{scene_name}_keyframes.json")
            
            scene_keyframes = {}
            for sensor, sequence in sensor_chains.items():
                if not sequence:
                    continue
                
                # 키프레임만 필터링
                keyframes = [entry for entry in sequence if entry['is_key_frame']]
                if keyframes:
                    scene_keyframes[sensor] = keyframes
            
            if scene_keyframes:
                with open(keyframe_file, 'w') as f:
                    json.dump(scene_keyframes, f, indent=2)
                
                total_keyframes = sum(len(frames) for frames in scene_keyframes.values())
                print(f"  {scene_name}: {total_keyframes}개 키프레임")
    
    def print_summary(self):
        """분석 결과 요약 출력"""
        print("\n" + "="*70)
        print("전체 분석 결과 요약")
        print("="*70)
        
        total_frames = 0
        total_keyframes = 0
        
        for sensor, chain in self.sensor_chains.items():
            if not chain:
                continue
                
            frames = len(chain)
            keyframes = len([entry for entry in chain if entry['is_key_frame']])
            total_frames += frames
            total_keyframes += keyframes
            
            print(f"{sensor:20}: {frames:6}개 프레임 (키프레임: {keyframes:4}개)")
        
        print("-"*70)
        print(f"{'총합':20}: {total_frames:6}개 프레임 (키프레임: {total_keyframes:4}개)")
        
        # Scene별 요약
        if self.scene_sensor_chains:
            print("\n" + "="*70)
            print("Scene별 분석 결과")
            print("="*70)
            
            for scene_name, sensor_chains in self.scene_sensor_chains.items():
                scene_total = sum(len(chain) for chain in sensor_chains.values())
                scene_keyframes = sum(len([entry for entry in chain if entry['is_key_frame']]) 
                                    for chain in sensor_chains.values())
                
                # Scene 정보 가져오기
                scene_info = None
                for scene in self.nusc.scene:
                    if scene['name'] == scene_name:
                        scene_info = scene
                        break
                
                print(f"\n{scene_name}:")
                if scene_info:
                    print(f"  설명: {scene_info['description']}")
                    print(f"  샘플 수: {scene_info['nbr_samples']}")
                
                sensor_count = len([s for s, c in sensor_chains.items() if c])
                print(f"  센서 종류: {sensor_count}개")
                print(f"  총 프레임: {scene_total}개 (키프레임: {scene_keyframes}개)")
            
            print(f"\n총 {len(self.scene_sensor_chains)}개 Scene 분석 완료")


def main():
    parser = argparse.ArgumentParser(description='NuScenes 데이터 프레임 순서 분석 도구 (NuScenes API 사용)')
    parser.add_argument('--dataroot', '-d', 
                       default='../data/nuscenes/raw',
                       help='NuScenes 데이터 루트 디렉토리')
    parser.add_argument('--version', '-v', 
                       default='v1.0-mini',
                       help='NuScenes 데이터 버전')
    parser.add_argument('--output', '-o', 
                       default='../output/nuscenes_analysis',
                       help='출력 디렉토리')
    parser.add_argument('--export', action='store_true',
                       help='센서별 순서 데이터를 JSON 파일로 저장')
    parser.add_argument('--keyframes', action='store_true',
                       help='키프레임 분석 결과를 별도로 저장')
    
    args = parser.parse_args()
    
    try:
        # 분석기 초기화 및 실행
        analyzer = NuScenesFrameAnalyzer(args.dataroot, args.version)
        analyzer.load_data()
        analyzer.analyze_all_scenes_and_sensors()
        
        if args.export:
            analyzer.export_sensor_sequences(args.output)
        
        if args.keyframes:
            analyzer.export_keyframe_analysis(args.output)
        
        analyzer.print_summary()
        
    except Exception as e:
        print(f"오류 발생: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
