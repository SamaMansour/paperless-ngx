// ...existing code...

// Ambient module/type declarations to satisfy the TypeScript compiler in the workspace
// (these are lightweight shims; prefer installing proper @types or the packages themselves)

declare const $localize: (messageParts: TemplateStringsArray, ...substitutions: any[]) => string;

declare module 'utif' {
  const anything: any
  export = anything
}

declare module 'ngx-device-detector' {
  export class DeviceDetectorService {
    isMobile(): boolean
    isDesktop(): boolean
    // add other commonly used methods if needed
  }
}

declare module '@ngneat/dirty-check-forms' {
  export function dirtyCheck(form: any, store$: any): any
  export interface DirtyComponent {}
}

// catch-all for packages without types
declare module '*'
